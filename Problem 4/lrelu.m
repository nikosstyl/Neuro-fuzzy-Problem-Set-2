function [out] = lrelu(in)

if in >= 0
    out = in;
else
    out = 0.001;
end