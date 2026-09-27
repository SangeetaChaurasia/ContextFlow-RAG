import logging
import time
import re

from hybrid_retriever import search
from context_builder import build_context
from reranker import rerank
from generator import generate_answer
from query_rewriter import rewrite_question
from query_classifier import classify_question

logger = logging.getLogger(__name__)

def retrieve_context(
    question,
    company_id,
    request_id=None,
    top_k=5,
    alpha=0.5,
    category=None,
    order_id=None,
    rerank_top_k=3,
    history=None,
):
    try:
        request_start = time.perf_counter()
        rewritten_question = question

        if history:
            rewrite_start = time.perf_counter()

            rewritten_question = rewrite_question(
                question,
                history,
            )

            rewrite_time = time.perf_counter() - rewrite_start

            logger.info(
                "[%s] Query rewritten in %.3fs | original=%s | rewritten=%s",
                request_id,
                rewrite_time,
                question,
                rewritten_question,
            )

        if order_id is None:
            order_match = re.search(
                r"\bORD-\d+\b",
                rewritten_question,
            )

            if order_match:
                order_id = order_match.group(0)


        logger.info(
            "[%s] Extracted order_id=%s",
            request_id,
            order_id,
        )

        query_type = classify_question(rewritten_question)

        category_map = {
            "order": "orders",
            "shipping": "shipping",
            "account": "account",
            "refund": "refund",
        }

        if category is None:
            category = category_map.get(query_type)

        logger.info(
            "[%s] Query classified as: %s | category filter: %s",
            request_id,
            query_type,
            category,
        )

        retrieval_start = time.perf_counter()

        results = search(
            question=rewritten_question,
            company_id=company_id,
            top_k=top_k,
            alpha=alpha,
            category=category,
            order_id=order_id,
        )

        retrieval_time = time.perf_counter() - retrieval_start

        logger.info(
            "[%s] Retrieval completed in %.3fs",
            request_id,
            retrieval_time,
        )

        logger.info(
            "[%s] Retrieved %d documents",
            request_id,
            len(results.matches),
        )

        logger.info(
            "[%s] Conversation history contains %d messages",
            request_id,
            len(history or []),
        )

        results_dict = {
            "matches": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                }
                for match in results.matches
            ]
        }

        rerank_start = time.perf_counter()

        reranked_results = rerank(
            question,
            results_dict,
            top_k=rerank_top_k,
        )

        rerank_time = time.perf_counter() - rerank_start

        logger.info(
            "[%s] Reranking completed in %.3fs",
            request_id,
            rerank_time,
        )

        logger.info(
            "[%s] Reranked %d documents",
            request_id,
            len(reranked_results),
        )

        if reranked_results:
            logger.info(
                "[%s] Top result: %s",
                request_id,
                reranked_results[0]["metadata"].get("title"),
            )

        if not reranked_results:
            return {
                "results": [],
                "context": "",
                "answer": "I don't have enough information to answer that.",
            }

        context = build_context(reranked_results)

        generation_start = time.perf_counter()

        answer = generate_answer(
            question,
            context,
        )

        generation_time = time.perf_counter() - generation_start

        logger.info(
            "[%s] Generation completed in %.3fs",
            request_id,
            generation_time,
        )

        total_time = time.perf_counter() - request_start

        logger.info(
            "[%s] Total RAG processing time: %.3fs",
            request_id,
            total_time,
        )

        return {
            "results": results,
            "context": context,
            "answer": answer,
        }

    except Exception as error:
        print(f"RAG service error: {error}")
        raise
