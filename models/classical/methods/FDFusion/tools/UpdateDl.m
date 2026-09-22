 
function  [ Dl ] = UpdateDl( Xl, Dl,Al, maxIter)
% ||J||_* s.t. X=XZ,Z=J;
[m1 n1] = size(Dl);
I =eye( size(Dl,2));
Y1=zeros(m1,n1);
mu1 = 0.5; % this one can be tuned
max_mu=10^30;
rho1=1.1;
epsilon1=1e-8;
MAX_ITER=maxIter;
iter=0;
while true
    if iter>MAX_ITER
        break;
    end
      Y=Dl+Y1/mu1;
    tau=1.0/mu1;
    [J]=singular_value_shrinkage(Y,tau);
      Dl=((Xl*Al')+ J+(1/mu1)*Y1)/(I+(Al*Al')); 
      Y1=Y1+mu1*(Dl-J);
    mu1=min(rho1*mu1,max_mu);
    if max(max(abs(Dl-J)))<epsilon1
        break;
    end
    iter=iter+1;
end

end
    


