# web-crawler-gpu
Experimental Web Crawler using GPU (CUDA + PyTorch + BERT)
Updated on 7/27/2023

This project is still in alpha stage.
It crawls the sites asynchronously and gathers the essential tokens (keywords) and stores it in JSON array into a CSV.

Tested to work on Ubuntu 20.04 with NVIDIA's CUDA 11.4 + GeForce RTX 2080Ti with a 10th generation Core i5 CPU with 12 threads.
Also tested to work on Macbook Pro with M1 Pro. 

Created mostly with the help of ChatGPT (GPT-4) but with several hours of work in improving and taking bugs out.

bert-crawler-hybrid5.py follows only the links listed in ursl.txt

gpu-crawler-recursive7.py is the recursive version of the web-crawler-gpu and starts from URLs in seeds.txt.

gpu-crawler-recursive12.py is the latest version with some debugging information added to gpu-crawler-recursive11.py.



New: 7/17/2023: I added a experimental folder with crawler-bert-embeddings.py inside. This outputs embeddings of each URLs crawled. Still experimental and needs code revision to be useful, but a reference for future use.

Update: 7/17/2023 17:33PM I added a experimental web crawler that outputs BERT embeddings using a very simple quantum algorithm using quantum entanglement. Tested to work on quantum simulator. The file name is: quantum-bert-crawler.py

gpu-crawler-recursive7.py running on NVIDIA GeForce RTX 2080Ti with 64GB of DDR4 RAM:

[![gpu-crawler-recursive7.py running on NVIDIA GeForce RTX 2080Ti:](https://img.youtube.com/vi/-9NsB_3lpRI/0.jpg)](https://www.youtube.com/watch?v=-9NsB_3lpRI)


gpu-crawler-recursive7.py running on M1 Pro:
[![gpu-crawler-recursive7.py running on M1 Pro:](https://img.youtube.com/vi/86yhWTWNWJM/0.jpg)](https://www.youtube.com/watch?v=86yhWTWNWJM)


(C)Tsubasa Kato 2023 - Inspire Search Corporation
https://www.inspiresearch.io/en
