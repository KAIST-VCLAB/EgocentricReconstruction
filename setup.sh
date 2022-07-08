rm -rf dmc
rm -rf RAFT
git clone https://github.com/princeton-vl/RAFT.git
git clone https://github.com/planaria/dmc.git
cp -Rf delta/dmc/* dmc
cd dmc
mkdir build
cd build
cmake ..
make
cd ../..
cp -Rf delta/raft/* RAFT
