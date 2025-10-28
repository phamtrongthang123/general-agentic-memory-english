#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM 框架 + LongMemEval 数据集测试文件

结合 run_generation.py 的数据处理逻辑和 GAM 框架，测试在长期记忆评估数据上的效果。
"""

import sys
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from gam import (
    MemoryAgent,
    ResearchAgent,
    VLLMGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)

# ========== 数据加载：借鉴自 run_generation.py ==========

def load_longmemeval(json_path: str) -> List[Dict[str, Any]]:
    """Load LongMemEval JSON and return the list of samples."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        # 如果是 JSONL 格式
        data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    return data

def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    """
    将 LongMemEval 的对话数据转换为 session 格式
    按照时间段和 session_id 组织对话，包含 Current Date 信息
    """
    sessions = []
    
    # 获取所有对话数据
    haystack_dates = sample.get("haystack_dates", [])
    haystack_session_ids = sample.get("haystack_session_ids", [])
    haystack_sessions = sample.get("haystack_sessions", [])
    
    # 按时间顺序组织对话
    for i, (date, session_id, session_content) in enumerate(zip(haystack_dates, haystack_session_ids, haystack_sessions)):
        # 构建 session 文本
        session_text = f"=== SESSION {i+1} - Date: {date} - Session ID: {session_id} ===\n"
        session_text += f"Current Date: {date}\n"
        session_text += "\n"
        
        # 添加对话内容
        for turn in session_content:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            session_text += f"{role}: {content}\n"
        
        sessions.append(session_text.strip())
    
    return sessions

def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 LongMemEval 样本中提取 QA 信息"""
    qas = []
    question_id = sample.get("question_id", "")
    question_type = sample.get("question_type", "unknown")
    
    qas.append({
        "question_id": question_id,
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "question_type": question_type,
        "question_date": sample.get("question_date"),
        "answer_session_ids": sample.get("answer_session_ids", []),
    })
    return qas

# ========== Prompt 设计：借鉴自 run_generation.py ==========

def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    """Try to parse a model's output (string or dict) into dict. Return None if fail."""
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    s = candidate.strip()
    l = s.find('{')
    r = s.rfind('}')
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l:r+1])
    except Exception:
        return None

def make_memory_only_prompt(memory_obj: Any, question: str, question_date: str) -> str:
    """基于记忆状态回答问题的 prompt：借鉴自 run_generation.py"""
    mem_str = json.dumps(memory_obj, ensure_ascii=False, indent=2) if isinstance(memory_obj, dict) else str(memory_obj)
    return f"""
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.

MEMORY STATE:
{mem_str}

Current Date: {question_date}
Question: {question}
Answer:
"""

def make_summary_prompt(summary: str, question: str, question_date: str) -> str:
    """基于研究摘要回答问题的 prompt：借鉴自 run_generation.py"""
    return f"""
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.

RESEARCH SUMMARY:
{summary}

Current Date: {question_date}
Question: {question}
Answer:
"""

def answer_with_summary(summary: str, question: str, question_date: str, generator) -> str:
    """基于研究摘要回答问题"""
    prompt = make_summary_prompt(summary, question, question_date)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()

def answer_with_memory(final_memory: Dict[str, Any], question: str, question_date: str, generator) -> str:
    """基于记忆状态回答问题"""
    prompt = make_memory_only_prompt(final_memory, question, question_date)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()

# ========== 核心处理逻辑 ==========

def process_sample(sample: Dict[str, Any], sample_index: int, outdir: str):
    """
    使用 GAM 框架处理单个样本。
    
    流程：
    1. 使用 MemoryAgent 构建记忆
    2. 使用 ResearchAgent 进行深度研究
    3. 基于研究结果进行问答
    """
    question_id = sample.get("question_id", f"sample_{sample_index}")
    
    print(f"\n{'='*60}")
    print(f"处理样本 #{sample_index}: {question_id}")
    print(f"{'='*60}")
    
    try:
        # 1. 构建会话块
        session_chunks = build_session_chunks_for_sample(sample)
        print(f"会话数: {len(session_chunks)}")
        if session_chunks:
            print(f"第一个会话预览:\n{session_chunks[0][:400]}...")
        
        # 创建输出目录
        sample_results_dir = os.path.join(outdir, question_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        print(f"输出目录: {sample_results_dir}")
        
        # 2. 创建共享存储
        memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
        page_store = InMemoryPageStore(dir_path=sample_results_dir)
        
        # 3. 创建 Generator
        print(f"\n步骤 1: 创建 Generator")
        generator_config = VLLMGeneratorConfig(
            model_name="qwen2.5-14b-instruct",
            api_key="empty",
            base_url="http://localhost:8000/v1",
            temperature=0.3,
            max_tokens=2048
        )
        generator = VLLMGenerator(generator_config.__dict__)
        print(f"[OK] Generator 创建完成")
        
        # 4. 使用 MemoryAgent 构建记忆（将每个 session 作为一条消息）
        print(f"\n步骤 2: 使用 MemoryAgent 构建记忆")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=generator
        )
        
        for i, session_chunk in enumerate(session_chunks, 1):
            print(f"  处理会话 {i}/{len(session_chunks)}...")
            memory_update = memory_agent.memorize(session_chunk)
        
        # 查看构建的记忆
        final_state = memory_store.load()
        print(f"[OK] 记忆构建完成！共 {len(final_state.abstracts)} 条记忆摘要")
        
        # 显示记忆摘要
        print("\n📚 记忆摘要:")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")
        
        # 保存记忆状态
        memory_state_file = os.path.join(sample_results_dir, "memory_state.json")
        with open(memory_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 记忆状态已保存: {memory_state_file}")
        
        # 5. 创建检索器
        print(f"\n步骤 3: 创建检索器")
        retrievers = {}
        
        # 索引检索器
        try:
            index_config = IndexRetrieverConfig(
                index_dir=os.path.join(sample_results_dir, "page_index")
            )
            index_retriever = IndexRetriever(index_config.__dict__)
            index_retriever.build(page_store)
            retrievers["page_index"] = index_retriever
            print(f"[OK] 索引检索器创建成功")
        except Exception as e:
            print(f"[WARN] 索引检索器创建失败: {e}")
        
        # BM25 检索器
        try:
            bm25_config = BM25RetrieverConfig(
                index_dir=os.path.join(sample_results_dir, "bm25_index"),
                threads=4
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            print(f"[OK] BM25 检索器创建成功")
        except Exception as e:
            print(f"[WARN] BM25 检索器创建失败: {e}")
        
        # Dense 检索器
        try:
            dense_config = DenseRetrieverConfig(
                index_dir=os.path.join(sample_results_dir, "dense_index"),
                model_name="/share/project/bingyu/models/bge-base-en-v1.5",
                devices=["cuda:0"]
            )
            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
            print(f"[OK] Dense 检索器创建成功")
        except Exception as e:
            print(f"[WARN] Dense 检索器创建失败: {e}")
        
        print(f"[INFO] 成功创建 {len(retrievers)} 个检索器")
        
        # 6. 创建 ResearchAgent
        print(f"\n步骤 4: 创建 ResearchAgent")
        research_agent = ResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=generator,
            max_iters=3
        )
        print(f"[OK] ResearchAgent 创建完成")
        
        # 7. 进行问答
        print(f"\n步骤 5: 进行问答")
        qas = collect_qa_items_for_sample(sample)
        print(f"共有 {len(qas)} 个问题需要回答")
        
        # 将记忆转换为字符串格式
        final_memory_str = json.dumps(final_state.model_dump(), ensure_ascii=False, indent=2)
        
        qa_results = []
        
        for i, qi in enumerate(qas, 1):
            q = qi.get("question") or ""
            gold = qi.get("answer")
            question_type = qi.get("question_type")
            question_date = qi.get("question_date")
            answer_session_ids = qi.get("answer_session_ids", [])
            
            print(f"\n--- 问题 {i}/{len(qas)} ---")
            print(f"问题: {q}")
            print(f"标准答案: {gold}")
            print(f"问题类型: {question_type}")
            print(f"问题日期: {question_date}")
            print(f"答案会话ID: {answer_session_ids}")
            
            try:
                # 使用 ResearchAgent 进行研究
                print("正在进行深度研究...")
                result = research_agent.research(q)
                research_summary = result.integrated_memory
                print(f"[OK] 研究完成！迭代次数: {len(result.raw_memory.get('iterations', []))}")
                print(f"研究摘要: {research_summary[:200]}...")
                
                # 保存研究轨迹
                research_trace = {
                    "question": q,
                    "raw_memory": result.raw_memory,
                    "integrated_memory": result.integrated_memory,
                    "iterations": result.raw_memory.get("iterations", []),
                    "search_plans": result.raw_memory.get("search_plans", []),
                    "reflections": result.raw_memory.get("reflections", [])
                }
                
                # 保存单个问题的研究轨迹
                trace_file = os.path.join(sample_results_dir, f"research_trace_q{i}.json")
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(research_trace, f, ensure_ascii=False, indent=2)
                print(f"[INFO] 研究轨迹已保存: {trace_file}")
                
                # 基于研究结果生成答案
                print("生成答案...")
                summary_answer = answer_with_summary(research_summary, q, question_date, generator)
                memory_answer = answer_with_memory(final_memory_str, q, question_date, generator)
                
                print(f"基于研究的答案: {summary_answer}")
                print(f"基于记忆的答案: {memory_answer}")
                
                qa_result = {
                    "question_id": qi.get("question_id"),
                    "question": q,
                    "gold_answer": gold,
                    "question_type": question_type,
                    "question_date": question_date,
                    "answer_session_ids": answer_session_ids,
                    "research_summary": research_summary,
                    "summary_answer": summary_answer,
                    "memory_answer": memory_answer,
                    "iterations": len(result.raw_memory.get("iterations", [])),
                    "research_trace_file": trace_file
                }
                qa_results.append(qa_result)
                
            except Exception as e:
                print(f"[ERROR] 处理问题失败: {e}")
                import traceback
                traceback.print_exc()
                qa_result = {
                    "question_id": qi.get("question_id"),
                    "question": q,
                    "gold_answer": gold,
                    "question_type": question_type,
                    "question_date": question_date,
                    "answer_session_ids": answer_session_ids,
                    "error": str(e)
                }
                qa_results.append(qa_result)
        
        # 保存结果
        results_file = os.path.join(sample_results_dir, "qa_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 结果已保存到: {results_file}")
        
        # 保存所有研究轨迹的汇总
        all_research_traces = []
        for i, qa_result in enumerate(qa_results, 1):
            if "research_trace_file" in qa_result:
                trace_file = qa_result["research_trace_file"]
                if os.path.exists(trace_file):
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        trace_data = json.load(f)
                        all_research_traces.append({
                            "question_index": i,
                            "question_id": qa_result["question_id"],
                            "question": qa_result["question"],
                            "question_type": qa_result["question_type"],
                            "research_trace": trace_data
                        })
        
        if all_research_traces:
            traces_summary_file = os.path.join(sample_results_dir, "all_research_traces.json")
            with open(traces_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_research_traces, f, ensure_ascii=False, indent=2)
            print(f"[OK] 所有研究轨迹汇总已保存到: {traces_summary_file}")
        
        # 总结
        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"问题ID: {question_id}")
        print(f"会话数: {len(session_chunks)}")
        print(f"记忆摘要数: {len(final_state.abstracts)}")
        print(f"处理问题数: {len(qa_results)}")
        print(f"研究轨迹文件数: {len(all_research_traces)}")
        print(f"结果保存到: {sample_results_dir}")
        print(f"  - QA结果: qa_results.json")
        print(f"  - 记忆状态: memory_state.json")
        print(f"  - 研究轨迹汇总: all_research_traces.json")
        print(f"  - 单个研究轨迹: research_trace_q*.json")
        
        return qa_results
        
    except Exception as e:
        error_msg = f"处理样本 {sample_index} 时出错: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return []


# ========== 主函数 ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GAM 框架 + LongMemEval 数据集测试")
    parser.add_argument("--data", type=str, default="/share/project/bingyu/datasets/longmemeval/longmemeval_s_cleaned.json", 
                        help="LongMemEval 数据集路径")
    parser.add_argument("--outdir", type=str, default="/share/project/bingyu/code/general-agentic-memory/results/longmemeval_output",
                        help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, help="开始样本索引")
    parser.add_argument("--end-idx", type=int, default=5, help="结束样本索引（不包含），None表示处理所有样本")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GAM 框架 + LongMemEval 数据集测试")
    print("=" * 60)
    print(f"数据集: {args.data}")
    print(f"输出目录: {args.outdir}")
    print(f"样本范围: {args.start_idx} 到 {args.end_idx-1 if args.end_idx else '全部'} (共 {args.end_idx - args.start_idx if args.end_idx else '全部'} 个样本)")
    print("=" * 60)
    
    # 加载数据
    samples = load_longmemeval(args.data)
    print(f"共加载 {len(samples)} 个样本")
    
    # 重新设置结束索引（在加载数据后）
    if args.end_idx is None:
        args.end_idx = len(samples)
    
    print(f"实际处理范围: {args.start_idx} 到 {args.end_idx-1} (共 {args.end_idx - args.start_idx} 个样本)")
    
    # 验证索引范围
    if args.start_idx < 0 or args.start_idx >= len(samples):
        print(f"错误: 开始样本索引 {args.start_idx} 超出范围 (总样本数: {len(samples)})")
        return
    
    if args.end_idx > len(samples):
        print(f"警告: 结束样本索引 {args.end_idx} 超出范围，调整为 {len(samples)}")
        args.end_idx = len(samples)
    
    if args.start_idx >= args.end_idx:
        print(f"错误: 开始索引 {args.start_idx} 必须小于结束索引 {args.end_idx}")
        return
    
    # 批量处理样本
    all_results = []
    for sample_idx in range(args.start_idx, args.end_idx):
        sample = samples[sample_idx]
        print(f"\n{'='*80}")
        print(f"开始处理样本 {sample_idx}/{len(samples)-1} (范围: {args.start_idx}-{args.end_idx-1})")
        print(f"{'='*80}")
        
        try:
            results = process_sample(sample, sample_idx, args.outdir)
            all_results.extend(results)
            print(f"[OK] 样本 {sample_idx} 处理完成")
        except Exception as e:
            print(f"[ERROR] 样本 {sample_idx} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存所有结果汇总
    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 批量结果汇总已保存: {summary_file}")
    
    print(f"\n{'='*60}")
    print("[OK] 批量测试完成！")
    print(f"处理样本数: {args.end_idx - args.start_idx}")
    print(f"成功处理: {len(all_results)} 个问题")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
