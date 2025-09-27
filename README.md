# ZigzagPointMamba: Spatial-Semantic Mamba for Point Cloud Understanding
<div align='center'>
    <a href='mailto:107552304043@stu.xju.edu.cn' target='_blank'>Linshuang Diao</a><sup>1</sup> 
    <a href='mailto:songsensen@stu.xju.edu.cn' target='_blank'>Sensen Song</a><sup>1</sup> 
    <a href='mailto:qyr@stu.xju.edu.cn' target='_blank'>Yurong Qian</a><sup>1</sup> 
    <a href='mailto:rdyedu@gmail.com' target='_blank'>Dayong Ren</a><sup>2</sup>
</div>
<div align='center'>
    <sup>1</sup>Xinjiang University  <sup>2</sup>Nanjing University 
</div>
<br>
<div align="center">
  <a href="https://Rabbitttttt218.github.io/ZigzagPointMamba/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=purple"></a>  
  <a href="https://arxiv.org/abs/2505.21381"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a>  
  <a href="https://github.com/Rabbitttttt218/ZigzagPointMamba"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue&logo=github"></a>  
  <a href="https://arxiv.org/pdf/2505.21381.pdf"><img src="https://img.shields.io/static/v1?label=PDF&message=Download&color=green"></a>  
</div>
<p align="center">
  <img src="ZigzagPointMamba_html/static/images/pipeline.png" height=400>
</p>


## Abstract
State Space models (SSMs) like PointMamba provide efficient feature extraction for point cloud self-supervised learning with linear complexity, surpassing Transformers in computational efficiency. However, existing PointMamba-based methods rely on complex token ordering and random masking, disrupting spatial continuity and local semantic correlations. We propose <strong>ZigzagPointMamba</strong> to address these challenges. The key to our approach is a simple zigzag scan path that globally sequences point cloud tokens, enhancing spatial continuity by preserving the proximity of spatially adjacent point tokens. Yet, random masking impairs local semantic modeling in self-supervised learning. To overcome this, we introduce a Semantic-Siamese Masking Strategy (SMS), which masks semantically similar tokens to facilitate reconstruction by integrating local features of original and similar tokens, thus overcoming dependence on isolated local features and enabling robust global semantic modeling. Our pre-training ZigzagPointMamba weights significantly boost downstream tasks, achieving a 1.59% mIoU gain on ShapeNetPart for part segmentation, a 0.4% higher accuracy on ModelNet40 for classification, and 0.19%, 1.22%, and 0.72% higher accuracies respectively for the classification tasks on the OBJ-BG, OBJ-ONLY, and PB-T50-RS subsets of ScanObjectNN.
## 🎉NEWS
+ [2025.09.19] 🎊 **Accepted to NeurIPS 2025**! Our paper *ZigzagPointMamba: Spatial-Semantic Mamba for Point Cloud Understanding* has been accepted to the 39th Annual Conference on Neural Information Processing Systems.
+ [2025.05.27] 🔥 Our paper "ZigzagPointMamba: Spatial-Semantic Mamba for Point Cloud Understanding" is available on arXiv! [arXiv:2505.21381](https://arxiv.org/abs/2505.21381)

## 📊 Experimental Results
### Key Visualizations
#### Zigzag Path & Masking
<p align="center"><img src="ZigzagPointMamba_html/static/images/zigzag_path_and_masking.png" height=250></p>3D zigzag scan path (spatial continuity) + SMS semantic masking.

#### Comprehensive Results
<p align="center"><img src="ZigzagPointMamba_html/static/images/comprehensive_results.png" height=250></p>(a) Cross-dataset performance; (b) SMS vs. random masking; (c) Feature fine-tuning effect.

#### ModelNet40 & ShapeNetPart
<p align="center"><img src="ZigzagPointMamba_html/static/images/Classification_on_ModelNet40_and_Part_Seg_on_ShapeNetPart.png" height=250></p>Classification (ModelNet40) and part segmentation (ShapeNetPart) results.

#### Few-shot Learning
<p align="center"><img src="ZigzagPointMamba_html/static/images/Few-shot.png" height=250></p>Superior few-shot classification performance on ModelNet40.

#### ScanObjectNN Results
<p align="center"><img src="ZigzagPointMamba_html/static/images/ScanobjNN.png" height=250></p>Consistent accuracy gains across all ScanObjectNN subsets.

## 📚 Citation
```bibtex
@inproceedings{diao2025zigzagpointmamba,
  title={ZigzagPointMamba: Spatial-Semantic Mamba for Point Cloud Understanding},
  author={Diao, Linshuang and Song, Sensen and Qian, Yurong and Ren, Dayong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
