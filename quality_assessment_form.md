# Quality assessment form

This file contains the questions used to initially assessed the ML models techincal gaps

1. Provide the model name(s) to be assessed with the information of this form.

2. Is the model part of a model family? A model is part of a family if it shares the same quality attributes, e.g.
   codebase, monitoring, retraining, etc. of the other models in the family.
   If so, please provide the name of the model family:

3. Link to the source code in git (if there are multiple code bases, please provide all the links)

4. Link to model documentation

5. Is the model deployed? Please provide the link in the ML registry.

6. Is the model used in production?

7. Please provide the link for the latest AB experiment.

8. What is (roughly) the average daily number of requests that the model receives?

9. Are there other teams or departments besides yours relying on this model?

10. What might be the consequences of disabling the model in production? Does this pose an existential risk to the
    Booking.com business?

11. Is the model being tested in an experiment (provide relevant link)?

12. Has the model been compared against a simple, low-cost, non-ML baseline? What is the relative improvement achieved?
    Please provide analysis supporting the claim

13. How many hours are needed for training? How many workflows are needed for the whole ML lifecycle?

14. How easy it is to go back to a previous model version in production? How can you achieve that?

15. How frequently have ML system failures happened in the model’s lifetime?

16. How frequently is the model retrained? Please provide any re-training pipeline link.

17. Does the model's source code (i.e. used for producing training data, model training, model evaluation, etc.) have
    automated tests? What is the test coverage?

18. How can you repeat the ML lifecycle (to deploy a new model version for example)? Is there any automation? What are
    the manual steps involved?

19. Which indicators are being monitored after the model is rolled out (e.g. model's performance, feature drift, feature
    parity, business metrics, distributions of input and outputs, etc.)? Please provide link to the relevant dashboard(
    s)

20. What are the latency and throughput requirements for the ML system? Are they met?

21. Which metadata and artifacts are logged during the ML lifecycle (e.g. datasets, hyper-parameters, evaluation
    metrics, model binary, etc) and are accessible by everyone in the team? How is this being done?

22. Is the output of the model stored in a table for consumption? If yes, please provide table's link

23. Did you use any explainability methods on your model? If yes, Which ones?

24. Is the model checked against any undesired biases (for more info see: go/ml-fairness) ?

25. Are there any standards to be met? Such as PII compliance? How are they met?

26. Do you perform input data validation (e.g. check for unexpected feature statistics, nulls and counts of input
    datasets, etc.)? If yes, what do you check and how?

27. Does the model consume user generated data or data from external sources (e.g. publicly available datasets
    or Any dataset downloaded outside the company ecosystem?

28. Do you filter out bots, to reduce the likelihood of the datasets being tampered intentionally? If yes, how?		