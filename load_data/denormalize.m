function x = normalize(x, mu, sigma)
	x=bsxfun(@times,x,sigma);
    x=bsxfun(@plus,x,mu);
end
