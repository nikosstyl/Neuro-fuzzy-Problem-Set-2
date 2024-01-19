function [out] = swish_deriv (in)
out = swish(in) + swish_sigma(in)*(1-swish(in));
end

function [out] = swish_sigma (in)
out = 1 / (1+exp(1)^(-in));
end