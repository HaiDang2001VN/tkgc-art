git clone https://github.com/HaiDang2001VN/tkgc-art.git src
git clone https://github.com/stmrdus/TKGC-Benchmark-Datasets.git dataset
mv dataset/Unified_Datasets src/data
cd src
pip install -r requirements.txt
pip install -r requirements_geometric.txt
g++ -std=c++23 -O3 -pthread -o prepare.o prepare.cpp
g++ -std=c++23 -O3 -pthread -o preprocess.o preprocess.cpp