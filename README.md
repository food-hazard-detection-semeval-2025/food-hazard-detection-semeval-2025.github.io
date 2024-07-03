# SemEval 2025 Task 9: The Food Hazard Detection Challenge

The Food Hazard Detection task evaluates explainable classification systems for titles of food-incident reports collected from the web. These algorithms may help automated crawlers find and extract food issues from web sources like social media in the future. Due to the potential high economic impact, transparency is crucial for this task.

![Example Data Overview](img/overview.png)
*Figure 1: The blue boxes are model inputs; the orange boxes are ground truth labels per sub-task. The number on the right indicates unique values per label.*

The SemEval-Task combines **two sub-tasks**:
- **(ST1)** Text classification for food hazard prediction, predicting the type of hazard and product.
- **(ST2)** Food hazard and product "vector" detection, predicting the exact hazard and product.

The task focuses on detecting the hazard and uses a two-step scoring metric based on the macro F1 score, focusing on the hazard label per sub-task.

# Task Organization

![Timeline](img/timeline.png)
*Figure 2: Challenge timeline: (a) training data available before the challenge; (b) validation data at the start, with gradual label release; (c) unlabeled test data for final ranking.*

The timeline is shown in Figure 2. Participants get training and validation data to build, train, and assess their systems before the evaluation period. The challenge takes place on Codalab and will be divided into **five phases** (all deadlines AoE time):

1. **Trial Phase** (*before September 2nd 2024*)
   - [labeled training data (5,082 samples)](data/incidents _training _full.csv) are available for devising and training models.

2. **Conception Phase ST1** (*September 2nd 2024 to October 14th 2024*)
   - Unlabeled validation data (565 samples) are released
   - Codalab accepts test submissions for ST1 (category classification). Only the predictions in **.csv** format will be uploaded.

3. **Conception Phase ST2** (*October 14th 2024 to January 10th 2025*)
   - Validation labels for ST1 are released
   - Codalab accepts test submissions for ST2 (vector classification). Only the predictions in **.csv** format will be uploaded.

4. **Evaluation Phase** (*January 10th 2025 to January 17th 2025*)
   - Validation labels for ST2 are released
   - Codalab accepts final submissions for both tasks. Only the predictions in **.csv** format will be uploaded.

5. **Paper Phase** (*January 17th 2025 to February 28th 2025*)
   - Labeled test data (997 samples) are released
   - Participants describe their systems in scientific papers and commit them to [SemEval 2025](https://semeval.github.io/SemEval2025/)

**Explainability** in food risk classification based on texts is currently underexplored although it may help humans quickly assess prediction validity and can be used for meta-learning approaches like clustering or pre-sorting examples. However, explanations can be diverse and task/model-dependent. Current literature includes both model-specific ([Assael et al., 2022](https://www.nature.com/articles/s41586-022-04448-z); [Pavlopoulos et al., 2022](https://aclanthology.org/2022.acl-long.259/)) and model agnostic ([Ribeiro et al., 2016](https://aclanthology.org/N16-3020/)) approaches. We aim to study mechanisms to explain decisions on food safety risks by asking participants to submit precise "vector"-labels (ST2) as explanations for their ST1 predictions.

Example code for this task can be found here after September 2nd 2024.

# Dataset

![Sample Texts and Labels](img/sample.png)
*Figure 3: Sample texts with labels.*

The dataset for this task consists of 6,644 short texts (length in characters: min=5, avg=88, max=277). Sample texts are shown in Figure 3. It includes manually labeled English food recall titles from official food agency websites (e.g., FDA). Each text is labeled by two food science or food technology experts. Upon task completion, the full dataset will be available under the Creative Commons BY-NC-SA 4.0 license on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10820657).

## The Ground Truth

Figure 3 shows a sample of the dataset. The data features "year," "month," "day," "language," "country," "title," and "text." Participants will base their analysis on either the "title" or the "text" feature (indicating which one they used). The task is to predict the labels "product-category" and "hazard-category" and the vectors "product" and "hazard." The class distribution is heavily imbalanced. The data includes 1,256 different products (e.g., "ice cream," "chicken based products," "cakes") sorted into 22 categories (e.g., "meat, egg and dairy products," "cereals and bakery products," "fruits and vegetables"). The 261 possible "hazard"-values (e.g., "salmonella," "listeria monocytogenes," "milk and products thereof") are sorted into 10 "hazard-category" values.

# Evaluation

We compute the performance for ST1 and ST2 by calculating the macro $\text{F} _1$-score on the participants' predicted labels $\hat{\bf{y}}$ using the annotated labels $\bf{y}$ as ground truth. This measure is the unweighted mean of per-class $\text{F} _1$-scores over the $n$ classes. Both $\hat{\bf{y}}$ and $\bf{y}$ are vectors of $m$ samples:

$\text{F} _1({\bf{y}}, \hat{{\bf{y}}}) = {\frac{2}{n} \sum} _{i=0}^{n} \frac{\text{RCL} _i ({\bf{y}}, \hat{{\bf{y}}}) \cdot \text{PRC} _i ({\bf{y}}, \hat{{\bf{y}}})}{\text{RCL} _i ({\bf{y}}, \hat{{\bf{y}}}) + \text{PRC} _i ({\bf{y}}, \hat{{\bf{y}}})}$

where $\text{RCL} _c$ is the recall and $\text{PRC} _c$ is the precision for a specific class $c$.
To combine the predictions for the hazard and product labels into one score, we take the average of the scores:

$\text{S}(Y, \hat{Y}) = \frac{\text{F} _1({\bf{y}}^{h}, \hat{{\bf{y}}}^{h}) + \text{F} _1({\bf{y}}^{p|h}, \hat{{\bf{y}}}^{p|h})}{2}$

Here $Y = \[{\bf{y}}^{h}, {\bf{y}}^{p}\]$ is the $2 \times m$ matrix with the hazard label $\bf{y} _h$ and the product label ${\bf{y}} _{p}$ as column vectors. The vector ${\bf{y}}^{p|h}$ is defined as the entries of ${\bf{y}}^{p}$ where ${\bf{y}}^{h}$ is correctly predicted:

${\bf{y}}^{p|h} = \{{\bf{y}} _{j}^{p} | \hat{\bf{y}} _{j}^{h}={{\bf{y}}} _{j}^{h}\}, j \in \{1,2,...,m\}$

The scalar ${\bf{y}} _{j}^{ * }$ is the $j$-th element of ${\bf{y}}^{ * }$. $\hat{Y}$ and $\hat{\bf{y}}^{p|h}$ are defined accordingly.
With this measure, we base our rankings predominantly on the predictions for the hazard classes. Intuitively, this means that a submission with both ${\bf{y}}^{h}$ and ${\bf{y}}^{p}$ completely right will score $1.0$, a submission with ${\bf{y}}^{h}$ completely right and ${\bf{y}}^{p}$ completely wrong will score $0.5$, and any submission with ${\bf{y}}^{h}$ completely wrong will score $0.0$ independently of the value of ${\bf{y}}^{p}$.

# Task Organizers

The organizers are:

- **Korbinian Randl** (lead) is a PhD student at Stockholm University. His research centers around machine learning for NLP and explainability. He works on applying these interests in food risk prediction.
- **John Pavlopoulos** (lead) is Assistant Professor at Athens University of Economics and Business and has co-organized BioASQ, HTREC, and SemEval Tasks 4 (2014) and 5 (2021). He is also affiliated with Stockholm University.
- **Aron Henriksson** is an associate professor at Stockholm University working on machine learning for NLP in healthcare and education.
- **Tony Lindgren** is Stockholm University's project leader for the EFRA project.
- **George Marinos** is a PhD student at Wageningen University & Research and AI Research Engineer in Agroknow, focusing on AI for forecasting food safety risks.
- **Manos Karvounis** is the research and innovation manager and the team leader of the R&D department in Agroknow, and the project coordinator of EFRA.

You can contact the organizers using [this email address](mailto:food-hazard-detection-semeval-2025@googlegroups.com).

# Ethical statement

All texts come from official and publicly available sources, so no privacy issues are present. All annotations are provided by Agroknow experts. The systems are intended to complement but not substitute human experts in preventing illness or harm from food sources.


![EFRA Funding](https://efraproject.eu/wp-content/uploads/2023/01/EFRA-logo-white-1-300x104.png)

*This challenge is part of the [EFRA project for Extreme Food Risk Analysis](https://efraproject.eu/) (funded by Horizon Europe under Grant Agreement No 101093026).*
