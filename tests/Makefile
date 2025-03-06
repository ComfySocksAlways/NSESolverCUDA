mtest:
	nvcc -O3 --std=c++20 -arch=sm_86 -I../src/ ../src/matrix.cu ../src/dmatrix.cu matrixtest.cu -o mtest.out

vtktest:
	nvcc -O3 --std=c++20 -arch=sm_86 -I../src/ ../src/matrix.cu ../src/printvtk.cu ../src/constants.cu vtktest.cu -o vtktest.out

gridtest:
	nvcc -O3 --std=c++20 -arch=sm_86 -I../src/ gridtest.cu ../src/matrix.cu ../src/grid.cu  -o gridtest.out

stenciltest:
	nvcc -O3 --std=c++20 -arch=sm_86 -I../src/ stenciltest.cu ../src/matrix.cu -o stenciltest.out

predtest:
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -I../src/ predtest.cu ../src/matrix.cu ../src/stencil.cu ../src/grid.cu
	nvcc -O3 --std=c++20 -arch=sm_86 -I../src/ predtest.o matrix.o stencil.o grid.o -o predtest.out
	rm -f predtest.o matrix.o stencil.o grid.o

rowsettest: 
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -I../src/ ../src/matrix.cu ../src/stencil.cu rowsettest.cu
	nvcc -O3 --std=c++20 -arch=sm_86 -I../src/ matrix.o stencil.o rowsettest.o -o rowsettest.out
	rm -f matrix.o stencil.o rowsettest.o

cusparsetest:
	nvcc -O3 --std=c++20 -arch=sm_86 -lcusparse cusparsetest.cu ../src/sparse.cu ../src/matrix.cu ../src/dvec.cu ../src/dmatrix.cu ../src/dscr_sys.cu -o cusparse.out

pressuretest:
	nvcc -O3 --std=c++20 -arch=sm_86 -lcusparse ../src/sparse.cu ../src/matrix.cu ../src/dvec.cu ../src/dmatrix.cu ../src/dscr_sys.cu ../src/pressure.cu ../src/constants.cu pressuretest.cu -o pressuretest.out

dottest: 
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse  -I../src/ ../src/solver.cu ../src/dvec.cu dottest.cu
	nvcc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ solver.o dvec.o dottest.o -o dottest.out
	rm -f solver.o dvec.o dottest.o

cgtest: sparse.o matrix.o solver.o dvec.o cgtest.o dscr_sys.o
	nvcc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ sparse.o matrix.o solver.o dvec.o cgtest.o dscr_sys.o -o cgtest.out

smoothtest: sparse.o matrix.o jacobi.o dvec.o smoothtest.o dscr_sys.o
	nvcc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ sparse.o matrix.o jacobi.o dvec.o smoothtest.o dscr_sys.o -o smoothtest.out

mgtest: sparse.o matrix.o jacobi.o dvec.o smoothtest.o dscr_sys.o multigrid.o mgtest.o
	nvcc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ sparse.o matrix.o jacobi.o dvec.o dscr_sys.o multigrid.o mgtest.o -o mgtest.out

psourcetest: sparse.o matrix.o solver.o dvec.o psourcetest.o dscr_sys.o dmatrix.o grid.o pressure.o stencil.o
	nvcc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ sparse.o matrix.o dmatrix.o grid.o pressure.o stencil.o solver.o dvec.o psourcetest.o dscr_sys.o -o psourcetest.out

sparse.o: ../src/sparse.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/sparse.cu -o sparse.o

matrix.o: ../src/matrix.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/matrix.cu -o matrix.o

dmatrix.o: ../src/dmatrix.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/dmatrix.cu -o dmatrix.o

grid.o: ../src/grid.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/grid.cu -o grid.o

solver.o: ../src/solver.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/solver.cu -o solver.o

jacobi.o: ../src/jacobi.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/jacobi.cu -o jacobi.o

multigrid.o: ../src/multigrid.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/multigrid.cu -o multigrid.o

dvec.o: ../src/dvec.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/dvec.cu -o dvec.o

cgtest.o: cgtest.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ cgtest.cu -o cgtest.o

smoothtest.o: smoothtest.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ smoothtest.cu -o smoothtest.o

mgtest.o: mgtest.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ mgtest.cu -o mgtest.o

psourcetest.o: psourcetest.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ psourcetest.cu -o psourcetest.o

dscr_sys.o: ../src/dscr_sys.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/dscr_sys.cu -o dscr_sys.o

pressure.o: ../src/pressure.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/pressure.cu -o pressure.o

stencil.o: ../src/stencil.cu
	nvcc -dc -O3 --std=c++20 -arch=sm_86 -lcusparse -I../src/ ../src/stencil.cu -o stencil.o


