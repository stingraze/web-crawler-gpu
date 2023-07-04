# web-crawler-gpu
Experimentary Web Crawler using GPU (CUDA + PyTorch + BERT)

This project is still in alpha stage.
It crawls the sites asynchronously and gathers the essential tokens (keywords) and stores it in JSON array into a CSV.

Tested to work on Ubuntu 20.04 with NVIDIA's CUDA 11.4 + GeForce RTX 2080Ti with a 10th generation Core i5 CPU with 12 threads.
Also tested to work on Macbook Pro with M1 Pro. 

Created mostly with the help of ChatGPT (GPT-4) but with several hours of work in improving and taking bugs out.

bert-crawler-hybrid5.py follows only the links listed in ursl.txt

gpu-crawler-recursive7.py is the recursive version of the web-crawler-gpu and starts from URLs in seeds.txt.

gpu-crawler-recursive7.py running on M1 Pro: (NVIDIA CUDA Demo video coming soon!)
[![gpu-crawler-recursive7.py running on M1 Pro:](https://img.youtube.com/vi/86yhWTWNWJM/0.jpg)](https://www.youtube.com/watch?v=86yhWTWNWJM)


(C)Tsubasa Kato 2023 - Inspire Search Corporation
https://www.inspiresearch.io/en
