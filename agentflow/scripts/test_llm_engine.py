import logging
import importlib
from typing import Any, List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Import the factory function
try:
    from agentflow.engine.factory import create_llm_engine
except Exception as e:
    logger.error(f"❌ Failed to import create_llm_engine: {e}")
    exit(1)


def test_engine_creation_robust(model_string: str, expected_class_name: str, **kwargs) -> Dict[str, Any]:
    """
    Robustly test engine creation: catch all errors and return status.
    """
    result = {
        "model_string": model_string,
        "expected": expected_class_name,
        "kwargs": kwargs,
        "success": False,
        "instance": None,
        "error": None
    }

    try:
        logger.info(f"🧪 Testing: '{model_string}' | kwargs={kwargs}")
        engine = create_llm_engine(model_string, **kwargs)
        class_name = engine.__class__.__name__

        if class_name != expected_class_name:
            error_msg = f"Expected {expected_class_name}, got {class_name}"
            logger.error(f"❌ {error_msg}")
            result["error"] = error_msg
            return result

        logger.info(f"✅ Success: Created {class_name}")
        result["success"] = True
        result["instance"] = engine
        return result

    except ImportError as e:
        msg = f"🚫 Module not installed: {e}"
        logger.warning(f"⚠️  {msg}")
        result["error"] = msg
        return result

    except KeyError as e:
        msg = f"🚫 API key not found in environment: {e}"
        logger.warning(f"⚠️  {msg}")
        result["error"] = msg
        return result

    except ConnectionError as e:
        msg = f"🚫 Connection failed (network/API down): {e}"
        logger.warning(f"⚠️  {msg}")
        result["error"] = msg
        return result

    except Exception as e:
        # Catch all other errors (auth, model not found, etc.)
        msg = f"💥 Unexpected error: {type(e).__name__}: {e}"
        logger.warning(f"⚠️  {msg}")
        result["error"] = msg
        return result


def test_all_engines_with_fault_tolerance():
    """
    Test all model providers with full error tolerance.
    Never raises an exception. Reports all results at the end.
    """
    test_cases: List[Dict] = [
        {"model_string": "gpt-4o", "expected": "ChatOpenAI", "kwargs": {}},
        {"model_string": "azure-gpt-4", "expected": "ChatAzureOpenAI", "kwargs": {}},
        {"model_string": "dashscope-qwen2.5-3b-instruct", "expected": "ChatDashScope", "kwargs": {}},
        {"model_string": "claude-3-5-sonnet", "expected": "ChatAnthropic", "kwargs": {}},
        {"model_string": "deepseek-chat", "expected": "ChatDeepseek", "kwargs": {}},
        {"model_string": "gemini-1.5-flash", "expected": "ChatGemini", "kwargs": {}},
        {"model_string": "grok", "expected": "ChatGrok", "kwargs": {}},
        {"model_string": "vllm-meta-llama/Llama-3-8b-instruct", "expected": "ChatVLLM", "kwargs": {}},
        {"model_string": "litellm-openai/gpt-4o", "expected": "ChatLiteLLM", "kwargs": {}},
        {"model_string": "together-meta-llama/Llama-3-70b-chat-hf", "expected": "ChatTogether", "kwargs": {}},
        {"model_string": "ollama-llama3", "expected": "ChatOllama", "kwargs": {}},
        # Invalid case
        {"model_string": "unknown-model-123", "expected": "Unknown", "kwargs": {}},
    ]

    results = []

    logger.info(f"🚀 Starting fault-tolerant test for {len(test_cases)} engines...")

    for case in test_cases:
        result = test_engine_creation_robust(
            model_string=case["model_string"],
            expected_class_name=case["expected"],
            **case["kwargs"]
        )
        results.append(result)

    # Final Summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info("=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Passed: {len(successful)}")
    for r in successful:
        logger.info(f"   • {r['model_string']} → {r['expected']}")

    logger.info(f"❌ Failed: {len(failed)}")
    for r in failed:
        logger.info(f"   • {r['model_string']} → {r['error']}")

    logger.info("=" * 60)
    if failed:
        logger.warning(f"💔 Some engines failed. This is expected if APIs are not configured.")
    else:
        logger.info("🎉 All engines initialized successfully!")

    return results


def example_generate_with_error_handling():
    """
    Example: Try to generate, with full error handling.
    """
    logger.info("🔤 Running generate() example with error handling...")
    try:
        model = create_llm_engine("gemini-2.5-flash", is_multimodal=True)
        response = model.generate(["Cities $A$ and $B$ are $45$ miles apart. Alicia lives in $A$ and Beth lives in $B$. Alicia bikes towards $B$ at 18 miles per hour. Leaving at the same time, Beth bikes toward $A$ at 12 miles per hour. How many miles from City $A$ will they be when they meet?"])
        logger.info(f"💬 Response: {response}")
    except Exception as e:
        logger.warning(f"⚠️  Generate failed: {type(e).__name__}: {e}")

def test_generation_for_all_engines(results: List[Dict[str, Any]], question: str = "which country's capital is Paris?"):
    logger.info("=" * 60)
    logger.info(f"🧠 Starting generate() test with question: '{question}'")
    logger.info("=" * 60)

    generate_results = []

    for result in results:
        if not result["success"]:
            continue 

        model_string = result["model_string"]
        engine = result["instance"]

        try:
            logger.info(f"💬 Testing generate() → '{model_string}'")
            response = engine.generate(question)

            if isinstance(response, dict) and "content" in response:
                content = response["content"]
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)

            logger.info(f"🟢 Success: {model_string} → {content[:100]}...") 
            generate_results.append({
                "model_string": model_string,
                "success": True,
                "response": content,
                "error": None
            })

        except NotImplementedError as e:
            error_msg = f"❌ Multimodal-only model used for text: {e}"
            logger.warning(f"⚠️  {model_string}: {error_msg}")
            generate_results.append({
                "model_string": model_string,
                "success": False,
                "response": None,
                "error": error_msg
            })

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.warning(f"⚠️  {model_string}: {error_msg}")
            generate_results.append({
                "model_string": model_string,
                "success": False,
                "response": None,
                "error": error_msg
            })

    successful = [r for r in generate_results if r["success"]]
    failed = [r for r in generate_results if not r["success"]]

    logger.info("=" * 60)
    logger.info("📝 GENERATE() TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Success: {len(successful)}")
    for r in successful:
        logger.info(f"   • {r['model_string']}")

    logger.info(f"❌ Failed: {len(failed)}")
    for r in failed:
        logger.info(f"   • {r['model_string']} → {r['error']}")

    return generate_results

if __name__ == "__main__":
    results = test_all_engines_with_fault_tolerance()

    generation_results = test_generation_for_all_engines(
        results=results,
        question="which country's capital is Paris?"
    )

    example_generate_with_error_handling()
    logger.info("🎉 Testing complete. Script did NOT crash despite errors.")