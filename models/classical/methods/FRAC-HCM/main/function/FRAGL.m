function u=FRAGL(A,M,N)
iterNum = 35;
rad = 1;
lambda1 = 1;
lambda2 = 1;
alpha = 0.5;
belta = 0.5; 
m1 = 1;
m2 = 1;
u = zeros(M,N);
u(:,:) = 0.3;
u(40:60,40:80) = 0.7;
[Ix,Iy] = gradient(double(A));
f = Ix.^2+Iy.^2;
g = 1./(1+f);    
diswght = disweight(rad);
saliency = imfilter(A,disweight(rad),'replicate'); 
for n=1:iterNum
    [u,~,~] = FRAGL_v1(double(saliency),u,diswght,lambda1,lambda2,alpha,belta,m1,m2,g);
end
if  u(1,1)> 0.97
    u=1-u; 
end    