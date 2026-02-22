INCLUDES='-I llvm-project/llvm/include -I llvm-project/build/include -I llvm-project/mlir/include -I llvm-project/build/tools/mlir/include -I circt/include'

find circt/lib/Analysis -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Conversion -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Dialect -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Firtool -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Reduce -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Scheduling -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Support -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Target -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Tools -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done

find circt/lib/Transforms -name '*.cpp' -type f | while read cpp
do
    echo "${cpp}"
    $CXX -I "$(dirname "${cpp}")" $INCLUDES -c -o "${cpp}.o" "${cpp}"
done