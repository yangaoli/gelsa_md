# GeLSA: a GPU-accelerated Local Similarity Analysis Tool

**Yang Li1, Shuaishuai Xu1, Yunhui Xiong1, Dongmei Ai3, Shengwei Hou2, Li Charlie Xia1,**  
Yang Li<sup>1</sup>, Shuaishuai Xu<sup>1</sup>, Yunhui Xiong<sup>1</sup>, Dongmei Ai<sup>3</sup>, Shengwei Hou<sup>2</sup>, Li Charlie Xia<sup>1,*</sup>

<sup>1</sup>Department of Statistics and Financial Mathematics, School of Mathematics, South China University of Technology, Guangzhou 510641, China  
<sup>2</sup>Department of Ocean Science & Engineering, Southern University of Science and Technology, Shenzhen 518055, China  
<sup>3</sup>Department X, Y, Shenzhen 518055, China  

<sup>*</sup>Corresponding author(s).  
E-mail: lcxia@scut.edu.cn (Li C. Xia).

## Abstract

We introduce GeLSA (GPU-accelerated extended Local Similarity Analysis). This novel multi-core accelerated computing tool enables local similarity analysis (LSA) for large-scale time series data in microbiome and environmental sciences. Compared to the previous most efficient LSA implementation (eLSA), GeLSA achieved approximately a 144-fold increase in computational efficiency on GPU machines.This is because GeLSA adapted the max sum subarray dynamical programming algorithm for LSA, allowing efficient core-level parallelisation to use modern CPU/GPU architectures. GeLSA also generally accelerates LSA-derived algorithms, including the local trend analysis (LTA), permutation-based MBBLSA, theory-based DDLSA and STLTA methods.As demonstrated by benchmarks, GeLSA maintained the accuracy of those methods while substantially improving their efficiency. Applied to a 72-hour hourly microbiome series tracking nearly thousands of marine microbes, GeLSA revealed intriguing dynamic co-occurrence networks of phytoplankton, bacteria, and viruses in Shenzhen’s Daya Bay. Overall, GeLSA is a versatile and fast tool for large-scale time series analysis, and we have made it freely available for academic use at http://github.com/labxscut/gelsa. 

## KEYWORDS: Local similarity analysis; GPU acceleration; Time series; Microbiome; Multi-core parallelisation

