from enum import Enum, unique

PRACTICES = [
    "logging of metadata and artifacts",
    "code versioning",
    "data versioning",
    "model versioning",
    "documentation",
    "ownership",
    "use of containarized environment",
    "model performance monitoring",
    "monitor feature parity",
    "monitor feature drift",
    "alerting",
    "automate the ML lifecycle",
    "automated tests",
    "fast development lifecycle feedback loops for easier debugging",
    "achieve feature parity",
    "keep historical features up to date",
    "remove redundant features",
    "dealing with feedback loos in experimentation",
    "peer code review",
    "offline metric correlates with production metric",
    "mimic production setting in offline evaluation",
    "compare with a baseline",
    "impact of model staleness is known",
    "code modularity and reusability",
    "establish clear success metric before model design",
    "prevent consequences of using a model as input to another",
    "error analysis",
    "keep model up to date by feeding with new labelled data points",
    "AB test all model changes",
    "do not test model which is not promising offline",
    "understand AB assumption violations and use alternatives",
    "unified environment for all lifecycle steps",
    "reuse data, supported internal tooling and solutions",
    "turn experimental code into production code",
    "shadow deployment",
    "canary deployment (gradual rollouts)",
]

QUALITY_ATTRIBUTES = [
    "accuracy",
    "effectiveness",
    "responsiveness",
    "reusability",
    "cost-effectiveness",
    "efficiency",
    "recoverability",
    "availability",
    "resilience",
    "adaptability",
    "scalability",
    "extensibility",
    "maintainability",
    "modularity",
    "testability",
    "deployability",
    "repeatability",
    "operability",
    "monitoring",
    "discoverability",
    "learnability",
    "readability",
    "traceability",
    "understandability",
    "usability",
    "debuggability",
    "explainability",
    "fairness",
    "ownership",
    "standards-compliance",
    "vulnerability",
]

PRACTICES_COLUMN_NAME = "Practice"
QUALITY_ATTRIBUTE_COLUMN_NAME = "quality sub-characteristic"
WEIGHTS_COLUMN_NAME = "weight (0:None, 4: Max)"
MEAN_WEIGHTS = "mean_weights"

DEPRECATED_SUBCHARS = {
    "reusability",
    "recoverability",
    "extensibility",
    "deployability",
    "learnability",
    "debuggability",
}

QUALITY_ASSESSMENT_PRACTICES = [
    "logging of metadata and artifacts",
    "code versioning",
    "documentation",
    "ownership",
    "model performance monitoring",
    "monitor feature parity",
    "monitor feature drift",
    "automate the ML lifecycle",
    "automated tests",
    "compare with a baseline",
    "code modularity and reusability",
    "keep model up to date by feeding with new labelled data points",
    "bot detection",
    "input data validation",
    "bias assessment",
    "bias mitigation",
    "Latency and Throughput are measured and requirements are defined",
    "Deploy in an accessible serving system or store output in a table for consumption",
    "optimize technical resources for training and inference",
    "Deploy the model in a highly available & scalable serving system",
    "Deploy the model in a serving system where it can be managed (disabled, uploaded, reverted)",
    "Monitor Business metrics",
    "Register the system in a centralized ML registry",
    "Variables, functions, classes have human readable and clear naming",
    "The code has a unified style (like PEP8)",
    "Use explainable models (such as linear) or explanation techniques like Shapley values or LIME",
]


@unique
class AssessmentPractices(str, Enum):
    all = "all"
    assessment = "assessment"
    non_assessment = "non-assessment"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
