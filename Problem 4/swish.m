function [out] = swish (in)

out = in / (1+exp(1)^(-in));

end