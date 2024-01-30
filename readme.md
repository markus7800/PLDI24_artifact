
# Installation

## Docker

Install [docker](https://www.docker.com).

```
docker build -t gmm .
docker run -it --name gmm --rm gmm
```

## Manual

Install Python with package `matplotlib`.

Install Julia with packages `Distributions`, `PyCall`, and `PyPlot`.

PyCall should point to the python version which has matplotlib installed.


# Usage

Run
```
julia gmm.jl
```
Result stored in `gmm_result.txt`.

For plotting run
```
julia plot_results.jl
```

To view plot on host machine run and different terminal.
```
docker cp gmm:/GMM/gmm_times.pdf . && open gmm_times.pdf
```