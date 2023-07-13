g++ -O3 -c -fPIC -std=c++11 qlt_reader.cc -o qlt_reader.o
g++ -O3 -shared -Wl,-soname,libqltreader.so -o libqltreader.so  qlt_reader.o

rm qlt_reader.o
mkdir -p ../bin
mv libqltreader.so ../bin
