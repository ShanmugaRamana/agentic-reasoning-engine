# train/meta_reasoner/weight_optimizer.py
from src.logger import logger

def optimize_weights(performance_data: dict):
    """
    Calculates normalized voting weights based on reasoner performance.

    Args:
        performance_data (dict): The accuracy scores of each reasoner per topic.

    Returns:
        dict: A dictionary of normalized weights for each reasoner per topic.
    """
    logger.info("Optimizing reasoner weights...")
    optimized_weights = {}
    for topic, accuracies in performance_data.items():
        total_accuracy = sum(accuracies.values())
        if total_accuracy == 0:
            # Avoid division by zero, assign equal weight
            num_reasoners = len(accuracies)
            normalized_weights = {reasoner: 1.0 / num_reasoners for reasoner in accuracies}
        else:
            normalized_weights = {
                reasoner: acc / total_accuracy for reasoner, acc in accuracies.items()
            }
        optimized_weights[topic] = normalized_weights
    
    logger.info("Weight optimization complete.")
    return optimized_weights