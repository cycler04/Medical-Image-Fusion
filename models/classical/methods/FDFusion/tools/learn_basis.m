function B = learn_basis(X, S, l2norm, Binit)


L = size(X,1);
N = size(X,2);
M = size(S, 1);

% tic
SSt = S*S';
XSt = X*S';

if exist('Binit', 'var')
    dual_lambda = diag(Binit\XSt - SSt);
    max(dual_lambda)
    min(dual_lambda)
else
    dual_lambda = 10*abs(rand(M,1)); % any arbitrary initialization should be ok.
end

c = l2norm^2;
trXXt = sum(sum(X.^2));

lb=zeros(size(dual_lambda));
options = optimset('Algorithm','trust-region-reflective','GradObj','on', 'Hessian','on','Display','off');


[x, fval, exitflag, output] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);

fval_opt = -0.5*N*fval;
dual_lambda= x;

Bt = (SSt+diag(dual_lambda)) \ XSt';
B_dual= Bt';
fobjective_dual = fval_opt;


B= B_dual;


return;


function [f,g,H] = fobj_basis_dual(dual_lambda, SSt, XSt, X, c, trXXt)
% Compute the objective function value at x
L= size(XSt,1);
M= length(dual_lambda);

SSt_inv = inv(SSt + diag(dual_lambda));


if L>M
   
    f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
    
else
    
    f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
end
f= -f;

if nargout > 1   
    
    g = zeros(M,1);
    temp = XSt*SSt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    
    if nargout > 2
        
        H = -2.*((temp'*temp).*SSt_inv);
        H = -H;
    end
end

return