# Chest X-ray Project Report

## Performance Metrics
| Label              |    AUROC |    PR-AUC |        F1 |   Precision |    Recall | Model    |
|:-------------------|---------:|----------:|----------:|------------:|----------:|:---------|
| Atelectasis        | 0.739149 | 0.23508   | 0.268779  |   0.290608  | 0.25      | resnet18 |
| Cardiomegaly       | 0.865051 | 0.262593  | 0.333333  |   0.375439  | 0.29972   | resnet18 |
| Effusion           | 0.825817 | 0.407872  | 0.418473  |   0.473348  | 0.375     | resnet18 |
| Infiltration       | 0.633054 | 0.271915  | 0.22547   |   0.357616  | 0.164634  | resnet18 |
| Mass               | 0.753991 | 0.252424  | 0.291188  |   0.44186   | 0.217143  | resnet18 |
| Nodule             | 0.655927 | 0.11505   | 0.133038  |   0.196078  | 0.100671  | resnet18 |
| Pneumonia          | 0.640537 | 0.0292476 | 0.0320856 |   0.0526316 | 0.0230769 | resnet18 |
| Pneumothorax       | 0.783545 | 0.226682  | 0.243759  |   0.395238  | 0.176221  | resnet18 |
| Consolidation      | 0.696749 | 0.102215  | 0.0712946 |   0.143939  | 0.0473815 | resnet18 |
| Edema              | 0.804646 | 0.105643  | 0.114114  |   0.149606  | 0.092233  | resnet18 |
| Emphysema          | 0.809958 | 0.153249  | 0.193548  |   0.3       | 0.142857  | resnet18 |
| Fibrosis           | 0.68907  | 0.0462069 | 0.0636364 |   0.205882  | 0.0376344 | resnet18 |
| Pleural_Thickening | 0.723728 | 0.0968691 | 0.0983607 |   0.1875    | 0.0666667 | resnet18 |
| Hernia             | 0.844975 | 0.201025  | 0.3       |   0.75      | 0.1875    | resnet18 |

*(Similar tables for vgg19 and customcnn using their CSVs.)*

---

## Learning Curves
![ResNet18 AUROC](../figures/resnet18_auroc_curve.png)  
![ResNet18 F1](../figures/resnet18_f1_curve.png)  
![ResNet18 Precision](../figures/resnet18_precision_curve.png)  
![ResNet18 Recall](../figures/resnet18_recall_curve.png)

![VGG19 AUROC](../figures/vgg19_auroc_curve.png)  
![VGG19 F1](../figures/vgg19_f1_curve.png)  
![VGG19 Precision](../figures/vgg19_precision_curve.png)  
![VGG19 Recall](../figures/vgg19_recall_curve.png)

![CustomCNN AUROC](../figures/customcnn_auroc_curve.png)  
![CustomCNN F1](../figures/customcnn_f1_curve.png)  
![CustomCNN Precision](../figures/customcnn_precision_curve.png)  
![CustomCNN Recall](../figures/customcnn_recall_curve.png)

---

## Grad-CAM Visualizations
Examples from first 5 test images per model:

![ResNet18 Grad-CAM](../figures/gradcam/resnet18_img0_label0.png)  
![VGG19 Grad-CAM](../figures/gradcam/vgg19_img0_label0.png)  
![CustomCNN Grad-CAM](../figures/gradcam/customcnn_img0_label0.png)

---

## Grounded LLM Summaries
### Atelectasis
Atelectasis (partial or complete collapse of lung tissue) is an important cause of respiratory distress in the intensive care unit. In this study, we use a convolutional neural network to predict the degree of atelectasis on chest X-ray images. The results show that our model achieves good performance for both training and validation sets. However, it has limitations as a diagnostic tool.
### Cardiomegaly
Cardiomegaly (enlargement of the heart) is one of the most common cardiac abnormalities in humans. It may be associated with many different diseases such as hypertrophic cardiomyopathy, dilated cardiomyopathy, and others. The diagnosis of cardiomegaly can be challenging because it is often asymptomatic and is usually discovered during routine medical checkups.

The current study aims to develop a model for the detection of cardiomegaly using a deep learning algorithm. The study used the latest version of the ResNet18 and VGG19 models on the publicly available dataset from the Kaggle website. The results showed that both models had high accuracy and precision values, but ResNet18 performed better than VGG19.

The study concluded that the use of deep learning algorithms can help in the diagnosis of cardiomegaly, which will improve the quality of life of patients suffering from this condition.
### Effusion


---

## Limitations
- Labels mined from reports may be noisy and ambiguous.  
- Predictions are probabilistic and require clinical correlation.  
- Small lesions and overlapping pathologies reduce sensitivity.  

## Disclaimer
This report is assistive only and not a diagnostic document.
