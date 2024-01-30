
import DelimitedFiles: readdlm

file_name = "gmm_result.txt"
data_sizes = collect(10:10:100)

result = readdlm(file_name)
naive_times = result[:,1]
while_times = result[:,2]
for_times = result[:,3]

# install PyPlot.jl to recreate plot.
import PyPlot
p = PyPlot.figure(figsize=(4,4))
PyPlot.plot(data_sizes, naive_times, label="naive")
PyPlot.plot(data_sizes, while_times, label="while loop opt.", linestyle="--")
PyPlot.plot(data_sizes, for_times, label="for loop opt.", linestyle="-.")
PyPlot.legend()
PyPlot.xlabel("N")
PyPlot.ylabel("time (s)")
PyPlot.tight_layout()
PyPlot.savefig("gmm_times.pdf")