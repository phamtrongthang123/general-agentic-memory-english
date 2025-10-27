#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM 框架测试文件

此文件用于测试 General Agentic Memory (GAM) 框架的核心功能：
1. MemoryAgent - 记忆构建
2. ResearchAgent - 深度研究
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gam import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    VLLMGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    OpenAIGeneratorConfig,
    VLLMGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
    BM25Retriever,
    DenseRetriever,
)

# 检查 BM25 和 Dense Retriever 是否可用
BM25_AVAILABLE = BM25Retriever is not None
DENSE_AVAILABLE = DenseRetriever is not None

if not BM25_AVAILABLE:
    print("[WARN] BM25Retriever 不可用（需要 pyserini 依赖）")
if not DENSE_AVAILABLE:
    print("[WARN] DenseRetriever 不可用（需要 FlagEmbedding 依赖）")


def test_complete_workflow():
    """测试完整的 GAM 工作流程"""
    print("=" * 60)
    print("GAM 完整工作流程测试")
    print("=" * 60)
    
    # 1. 创建共享的存储实例
    print("步骤 1: 创建共享存储实例")
    memory_store = InMemoryMemoryStore(dir_path="/share/project/bingyu/code/general-agentic-memory/gam/test_memory_output")
    page_store = InMemoryPageStore(dir_path="/share/project/bingyu/code/general-agentic-memory/gam/test_memory_output")
    print("[OK] 存储实例创建完成")
    
    # 2. 创建 LLM Generator
    print("\n步骤 2: 创建 LLM Generator")
    # generator_config = OpenAIGeneratorConfig(
    #     model_name="gpt-4o-mini",
    #     api_key="sk-UdTVN7RUnJY0jMVM2aUMhSJKGu6nmwYDprWkEltPuDbxMuCR",
    #     base_url="https://api2.aigcbest.top/v1",
    #     temperature=0.3,
    #     max_tokens=200
    # )

    generator_config = VLLMGeneratorConfig(
        model_name="qwen2.5-14b-instruct",
        api_key="empty",
        base_url="http://localhost:8000/v1",
        temperature=0.3,
        max_tokens=2048
    )
    
    generator = VLLMGenerator(generator_config.__dict__)
    print("[OK] Generator 创建完成")
    
    # 3. 创建 MemoryAgent（使用共享存储）
    print("\n步骤 3: 创建 MemoryAgent")
    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=generator
    )
    print("[OK] MemoryAgent 创建完成")
    
    # 4. 构建记忆 - 处理一系列消息
    print("\n步骤 4: 构建记忆")
    messages = [
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是 AI 的一个子集，它使计算机能够在没有明确编程的情况下学习。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。",
        "自然语言处理（NLP）是 AI 的另一个重要分支，专注于使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是 AI 的另一个关键领域，致力于使计算机能够看和理解视觉信息。"
    ]
    
    print(f"处理 {len(messages)} 条消息...")
    for i, message in enumerate(messages, 1):
        print(f"  处理消息 {i}/{len(messages)}: {message[:50]}...")
        memory_update = memory_agent.memorize(message)
        print(f"  [OK] 记忆更新完成，当前记忆条数: {len(memory_update.new_state.abstracts)}")
    
    # 5. 查看记忆构建结果
    final_state = memory_store.load()
    print(f"\n[OK] 记忆构建完成！共 {len(final_state.abstracts)} 条记忆摘要")
    print("记忆摘要:")
    for i, abstract in enumerate(final_state.abstracts, 1):
        print(f"  {i}. {abstract}")
    
    # 6. 创建检索器
    print(f"\n步骤 5: 创建检索器")
    retrievers = {}
    
    # 索引检索器（不需要额外依赖）
    try:
        index_config = IndexRetrieverConfig(
            index_dir="/share/project/bingyu/code/general-agentic-memory/gam/test_memory_output/page_index"
        )
        index_retriever = IndexRetriever(index_config.__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
        print("[OK] 索引检索器已创建")
    except Exception as e:
        print(f"[WARN] 索引检索器创建失败: {e}")
    
    # BM25 检索器
    if BM25_AVAILABLE:
        try:
            print("\n尝试创建 BM25 检索器...")
            bm25_config = BM25RetrieverConfig(
                index_dir="/share/project/bingyu/code/general-agentic-memory/gam/test_memory_output/bm25_index",
                threads=4
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            print("[OK] BM25 检索器已创建")
        except Exception as e:
            print(f"[WARN] BM25 检索器创建失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[INFO] BM25 检索器不可用，跳过")
    
    # Dense 检索器
    if DENSE_AVAILABLE:
        try:
            print("\n尝试创建 Dense 检索器...")
            # 使用本地模型 all_minilm_l6_v2
            dense_config = DenseRetrieverConfig(
                index_dir="/share/project/bingyu/code/general-agentic-memory/gam/test_memory_output/dense_index",
                model_name="/share/project/bingyu/models/bge-base-en-v1.5",
                devices=["cuda:0"]
            )
            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
            print("[OK] Dense 检索器已创建")
        except Exception as e:
            print(f"[WARN] Dense 检索器创建失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[INFO] Dense 检索器不可用，跳过")
    
    print(f"\n[INFO] 成功创建 {len(retrievers)} 个检索器: {list(retrievers.keys())}")
    
    # 7. 创建 ResearchAgent（使用相同的存储和生成器）
    print(f"\n步骤 6: 创建 ResearchAgent")
    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=generator,
        max_iters=2
    )
    print("[OK] ResearchAgent 创建完成")
    
    # 8. 进行深度研究
    print(f"\n步骤 7: 进行深度研究")
    question = "机器学习和深度学习有什么区别？"
    print(f"研究问题: {question}")
    print("正在进行深度研究...")
    
    try:
        result = research_agent.research(question)
        print("\n[OK] 研究完成！")
        print(f"\n研究结果:\n{result.integrated_memory}")
        print(f"\n迭代次数: {len(result.raw_memory.get('iterations', []))}")
        
        # 9. 展示完整流程的结果
        print(f"\n步骤 8: 流程总结")
        print("=" * 40)
        print("📊 最终状态:")
        print(f"  - 记忆摘要数量: {len(memory_store.load().abstracts)}")
        print(f"  - 页面数量: {len(page_store.load())}")
        print(f"  - 检索器数量: {len(retrievers)}")
        print(f"  - 研究迭代次数: {len(result.raw_memory.get('iterations', []))}")
        
    except Exception as e:
        print(f"\n[ERROR] 研究失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[OK] 完整工作流程测试完成\n")
    return memory_agent, research_agent, memory_store, page_store



def test_load_existing_memory_workflow():
    """测试加载已有记忆的工作流程"""
    print("=" * 60)
    print("加载已有记忆工作流程测试")
    print("=" * 60)
    
    # 1. 创建存储实例（指向已有的记忆数据）
    print("步骤 1: 加载已有记忆数据")
    memory_store = InMemoryMemoryStore(dir_path="./test_memory_output")
    page_store = InMemoryPageStore(dir_path="./test_memory_output")
    
    # 加载记忆状态
    memory_state = memory_store.load()
    pages = page_store.load()
    
    print(f"[OK] 记忆加载完成！")
    print(f"  - 记忆摘要数量: {len(memory_state.abstracts)}")
    print(f"  - 页面数量: {len(pages)}")
    
    # 显示现有记忆摘要
    print("\n📚 现有记忆摘要:")
    for i, abstract in enumerate(memory_state.abstracts, 1):
        if abstract != "NO NEW INFORMATION":
            print(f"  {i}. {abstract}")
    
    # 2. 创建 LLM Generator
    print(f"\n步骤 2: 创建 LLM Generator")
    generator_config = VLLMGeneratorConfig(
        model_name="qwen2.5-14b-instruct",
        api_key="empty",
        base_url="http://localhost:8000/v1",
        temperature=0.3,
        max_tokens=2048
    )
    
    generator = VLLMGenerator(generator_config.__dict__)
    print("[OK] Generator 创建完成")
    
    # 3. 创建检索器
    print(f"\n步骤 3: 创建检索器")
    retrievers = {}
    
    # 索引检索器
    try:
        index_config = IndexRetrieverConfig(
            index_dir="./test_memory_output/page_index"
        )
        index_retriever = IndexRetriever(index_config.__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
        print("[OK] 索引检索器已创建")
    except Exception as e:
        print(f"[WARN] 索引检索器创建失败: {e}")
    
    # BM25 检索器
    if BM25_AVAILABLE:
        try:
            print("\n尝试创建 BM25 检索器...")
            bm25_config = BM25RetrieverConfig(
                index_dir="./test_memory_output/bm25_index",
                threads=4
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["bm25"] = bm25_retriever
            print("[OK] BM25 检索器已创建")
        except Exception as e:
            print(f"[WARN] BM25 检索器创建失败: {e}")
    else:
        print("[INFO] BM25 检索器不可用，跳过")
    
    # Dense 检索器
    if DENSE_AVAILABLE:
        try:
            print("\n尝试创建 Dense 检索器...")
            # 使用本地模型
            dense_config = DenseRetrieverConfig(
                index_dir="./test_memory_output/dense_index",
                model_name="/share/project/bingyu/models/all_minilm_l6_v2",
                devices="cuda"  # 可以改成 cuda 如果你有 GPU
            )
            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["dense"] = dense_retriever
            print("[OK] Dense 检索器已创建")
        except Exception as e:
            print(f"[WARN] Dense 检索器创建失败: {e}")
    else:
        print("[INFO] Dense 检索器不可用，跳过")
    
    print(f"\n[INFO] 成功创建 {len(retrievers)} 个检索器: {list(retrievers.keys())}")
    
    # 4. 创建 ResearchAgent
    print(f"\n步骤 4: 创建 ResearchAgent")
    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=generator,
        max_iters=3
    )
    print("[OK] ResearchAgent 创建完成")
    
    # 5. 进行多个研究测试
    print(f"\n步骤 5: 进行多个研究测试")
    research_questions = [
        "人工智能的主要分支有哪些？",
        "机器学习和深度学习的关系是什么？",
        "自然语言处理和计算机视觉有什么共同点？",
        "AI 系统如何模拟人类智能？"
    ]
    
    results = []
    for i, question in enumerate(research_questions, 1):
        print(f"\n--- 研究问题 {i}/{len(research_questions)} ---")
        print(f"问题: {question}")
        print("正在进行研究...")
        
        try:
            result = research_agent.research(question)
            results.append((question, result))
            print(f"[OK] 研究完成！")
            print(f"迭代次数: {len(result.raw_memory.get('iterations', []))}")
            print(f"研究结果摘要: {result.integrated_memory[:100]}...")
            
        except Exception as e:
            print(f"[ERROR] 研究失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. 展示研究结果
    print(f"\n步骤 6: 研究结果总结")
    print("=" * 50)
    print("🔍 研究结果详情:")
    
    for i, (question, result) in enumerate(results, 1):
        print(f"\n{i}. 问题: {question}")
        print(f"   结果: {result.integrated_memory}")
        print(f"   迭代次数: {len(result.raw_memory.get('iterations', []))}")
    
    # 7. 测试记忆扩展功能
    print(f"\n步骤 7: 测试记忆扩展功能")
    print("添加新的信息到现有记忆中...")
    
    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=generator
    )
    
    new_messages = [
        "强化学习是机器学习的另一个重要分支，通过奖励和惩罚机制来训练智能体。",
        "生成式AI是近年来发展迅速的技术，能够创建新的内容，如文本、图像和代码。"
    ]
    
    for i, message in enumerate(new_messages, 1):
        print(f"  添加新信息 {i}/{len(new_messages)}: {message[:50]}...")
        memory_update = memory_agent.memorize(message)
        print(f"  [OK] 记忆更新完成")
    
    # 8. 最终状态展示
    final_state = memory_store.load()
    print(f"\n步骤 8: 最终状态展示")
    print("=" * 40)
    print("📊 最终统计:")
    print(f"  - 总记忆摘要数量: {len(final_state.abstracts)}")
    print(f"  - 总页面数量: {len(page_store.list_all())}")
    print(f"  - 成功研究数量: {len(results)}")
    print(f"  - 检索器数量: {len(retrievers)}")
    
    print(f"\n📚 更新后的记忆摘要:")
    for i, abstract in enumerate(final_state.abstracts, 1):
        if abstract != "NO NEW INFORMATION":
            print(f"  {i}. {abstract}")
    
    print(f"\n[OK] 加载已有记忆工作流程测试完成！")
    return research_agent, memory_store, page_store, results


def test_additional_research():
    """测试额外的研究功能"""
    print("=" * 60)
    print("额外研究测试")
    print("=" * 60)
    
    # 这个函数展示如何基于已有的记忆进行新的研究
    print("[INFO] 提示: 在实际应用中，您可以:")
    print("  1. 基于已构建的记忆进行多次研究")
    print("  2. 根据研究结果更新和扩展记忆")
    print("  3. 使用不同的检索策略组合")
    print("  4. 调整研究参数（max_iters, temperature 等）")
    
    print("\n[OK] 额外研究测试说明完成\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("GAM 框架测试套件")
    print("=" * 60)
    print("\n请选择测试模式:")
    print("  1. 完整工作流程测试 (构建新记忆)")
    print("  2. 加载已有记忆测试 (基于现有记忆进行研究)")
    print("  3. 运行所有测试")
    print()
    
    try:
        import sys
        if len(sys.argv) > 1:
            choice = sys.argv[1]
        else:
            choice = input("请输入选择 (1/2/3): ").strip()
        
        if choice == "1":
            print("\n" + "=" * 60)
            print("运行完整工作流程测试")
            print("=" * 60)
            print("\n此测试将展示完整的 GAM 工作流程:")
            print("  1. 创建共享存储实例")
            print("  2. 创建 LLM Generator")
            print("  3. 创建 MemoryAgent 并构建记忆")
            print("  4. 创建检索器")
            print("  5. 创建 ResearchAgent 并进行研究")
            print("  6. 展示完整流程结果")
            print()
            
            # 运行完整工作流程测试
            memory_agent, research_agent, memory_store, page_store = test_complete_workflow()
            
            print("=" * 60)
            print("[OK] 完整工作流程测试完成！")
            print("=" * 60)
            print("\n📁 生成的文件:")
            print("  - ./test_memory_output/memory_state.json  (记忆状态)")
            print("  - ./test_memory_output/pages.json         (页面数据)")
            print("  - ./test_memory_output/page_index/         (页面索引)")
            
        elif choice == "2":
            print("\n" + "=" * 60)
            print("运行加载已有记忆测试")
            print("=" * 60)
            print("\n此测试将基于现有记忆进行:")
            print("  1. 加载已有记忆数据")
            print("  2. 创建研究代理")
            print("  3. 进行多个研究测试")
            print("  4. 测试记忆扩展功能")
            print("  5. 展示研究结果")
            print()
            
            # 运行加载已有记忆测试
            research_agent, memory_store, page_store, results = test_load_existing_memory_workflow()
            
            print("=" * 60)
            print("[OK] 加载已有记忆测试完成！")
            print("=" * 60)
            print(f"\n📊 测试结果:")
            print(f"  - 成功研究数量: {len(results)}")
            print(f"  - 最终记忆数量: {len(memory_store.load().abstracts)}")
            print(f"  - 页面数量: {len(page_store.load())}")
            
        elif choice == "3":
            print("\n" + "=" * 60)
            print("运行所有测试")
            print("=" * 60)
            
            # 运行完整工作流程测试
            print("\n--- 第一部分: 完整工作流程测试 ---")
            memory_agent, research_agent, memory_store, page_store = test_complete_workflow()
            
            # 运行加载已有记忆测试
            print("\n--- 第二部分: 加载已有记忆测试 ---")
            research_agent2, memory_store2, page_store2, results = test_load_existing_memory_workflow()
            
            # 额外研究测试说明
            test_additional_research()
            
            print("=" * 60)
            print("[OK] 所有测试完成！")
            print("=" * 60)
            print(f"\n📊 最终统计:")
            print(f"  - 记忆摘要数量: {len(memory_store2.load().abstracts)}")
            print(f"  - 页面数量: {len(page_store2.list_all())}")
            print(f"  - 研究测试数量: {len(results)}")
            
        else:
            print("无效选择，请运行 'python test_gam_framework.py 1' 或 'python test_gam_framework.py 2'")
            return
        
        print("\n[INFO] 使用建议:")
        print("  - 查看生成的文件了解数据格式")
        print("  - 可以基于现有记忆进行更多研究")
        print("  - 可以添加更多消息来扩展记忆")
        print("  - 可以调整检索器配置优化性能")
        print("  - 使用 'python test_gam_framework.py 2' 快速测试已有记忆")
        
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

