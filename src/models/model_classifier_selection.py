from src.models.edge_classifier.simple_edge_classifier import get_model as simple_classifier

def get_edge_classifier(model_config, in_features):
    match model_config["name"]:
        case "SimpleEdgeClassifier":
            return simple_classifier(model_config, in_features)
        case _:
            raise NotImplementedError("Edge Classifier model not implemented!")