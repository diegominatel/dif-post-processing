# Official DIF-PP Repository

DIF-PP is a post-processing method based on Item Response Theory (IRT) and Differential Item Functioning (DIF) for promoting group fairness in binary classifiers.

We first introduced the method in the paper entitled "A DIF-Driven Threshold Tuning Method for Improving Group Fairness", published in the proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing (SAC'25). We later expanded this work into an extended version, which we published in Applied Computing Review under the title "DIF-PP: Threshold Optimization Informed by IRT Models for Group Fairness in Machine Learning".

## BibTeX Citation

If you use any part of this code in your research, please cite it using the following BibTex entry:

```latex
@inproceedings{minatel2025dif,
  title={A DIF-Driven Threshold Tuning Method for Improving Group Fairness},
  author={Minatel, Diego and Parmezan, Antonio RS and Roque dos Santos, Nicolas and C{\'u}ri, Mariana and Lopes, Alneu},
  booktitle={Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
  pages={890--898},
  year={2025}
}
```

and

```latex
will be available soon
```

## Abstract of the paper: A DIF-Driven Threshold Tuning Method for Improving Group Fairness

To promote social good, current decision support systems based on machine learning must not propagate society's various types of discrimination. Consequently, a desirable behavior for classifiers used in decision-making is that their results do not favor or disadvantage any specific sociodemographic group. One way to achieve this behavior is through post-processing methods, which apply threshold tuning to select the decision boundary that enhances the impartiality of the trained model's decisions. Various strategies have been proposed to determine the optimal threshold, but finding the trade-off between fairness and predictive performance remains challenging. Recently, the application of Differential Item Functioning (DIF) concepts has proven effective for this purpose in model selection, which is a similar application. This finding makes using DIF in threshold tuning appealing and an unexplored contribution to the literature on fairness in machine learning. This paper addresses this gap and proposes DIF-PP, a novel post-processing method based on DIF. We experimentally evaluated our method against three baselines using fifteen datasets, six classification algorithms with sixteen settings for each one, four group fairness metrics, one predictive performance measure, one multi-criteria measure, and one statistical significance test. Our experimental results indicate that DIF-PP provides the best trade-off between group fairness metrics and predictive performance, making it the optimal choice for threshold tuning of binary classifiers applied to decision-making tasks involving people.

## Full texts:

- A DIF-Driven Threshold Tuning Method for Improving Group Fairness: https://dl.acm.org/doi/10.1145/3672608.3707875
- DIF-PP: Threshold Optimization Informed by IRT Models for Group Fairness in Machine Learning: will be available soon
