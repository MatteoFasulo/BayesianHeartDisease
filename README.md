# Bayesian Network for Heart Disease Risk Assessment

This repository encompasses the code and report for the "Fundamentals of Artificial Intelligence and Knowledge Representation (Mod. 3)" course at Alma Mater Studiorum Università di Bologna.

## Authors

- [Matteo Fasulo](https://github.com/MatteoFasulo)
- [Luca Tedeschini](https://github.com/LucaTedeschini)
- [Antonio Gravina](https://github.com/GravAnt)
- [Luca Babboni](https://github.com/ElektroDuck)

## Abstract

Cardiovascular disease (CVD) remains a significant cause of mortality in Europe, imposing both health and economic challenges. Timely and accurate prediction is crucial for effective prevention and intervention strategies. Identifying modifiable and non-modifiable risk factors is essential, as lifestyle changes can significantly impact individual health.

Bayesian networks (BNs) have emerged as valuable tools in healthcare for handling complex data and analyzing interactions among various risk factors. They've proven successful in assessing CVD risk, aiding real-time diagnosis, and predicting hidden patient conditions.

Our work, inspired by [Ordovas et al. (2023)](https://doi.org/10.1016/j.cmpb.2023.107405), aimed to replicate their BN-based CVD risk prediction using a different dataset. Additionally, we sought to explore the broader potential of BNs in CVD risk assessment, conducting in-depth analyses beyond the original paper.

## Dashboard

The dashboard is a web application enabling users to interact with the Bayesian Network. Users input variable values, and the dashboard calculates the probability of heart disease. Built using the Streamlit library, the required packages can be installed using:

```bash
pip install -U -r requirements.txt
```

Run the dashboard with:

```bash
streamlit run app.py
```

> The Web App is publicly available at [heart-disease-risk.streamlit.app](https://heart-disease-risk.streamlit.app)

## Source

Datasets used are accessible in the UCI Machine Learning Repository's Index of heart disease datasets: [UCI Heart Disease Datasets](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/)

## Dataset

>fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved March 2024 from [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction).

## Acknowledgements

1. Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

## References

[1] Wilkins, E., et al. (2017). European Cardiovascular Disease Statistics 2017. European Heart Network. [CVD Statistics Report](http://www.ehnheart.org/images/CVD-statistics-report-August-2017.pdf)

[2] Mahmood, S. S., et al. (2014). The Framingham Heart Study and the epidemiology of cardiovascular disease: a historical perspective. Lancet (London, England), 383(9921), 999–1008. [DOI](https://doi.org/10.1016/S0140-6736(13)61752-3)

[3] WHO CVD Risk Chart Working Group (2019). World Health Organization cardiovascular disease risk charts: revised models to estimate risk in 21 global regions. The Lancet. Global health, 7(10), e1332–e1345. [DOI](https://doi.org/10.1016/S2214-109X(19)30318-3)

[4] Jensen, Finn & Nielsen, Thomas. (2007). Bayesian Network and Decision Graphs. [DOI](https://doi.org/10.1007/978-0-387-68282-2).

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## License

This project is licensed under the [MIT License](LICENSE).
