function [ D0 ] = UpadteD0( PHH,Dhs,Dhl,Zls,Zll,H)

D0=zeros(size(Dhs));
for i=1:15
XX=[PHH-Dhs*H*Zls-Dhl*H*Zll;D0*H*Zls];
[d,n]=size(PHH);
IMat=eye(d,d);
I_M=[IMat;IMat];

X0 = UpdateX0( XX,I_M,0.0001 );
E1=eye(size(H));

D0=learn_basis(X0, H*Zls, 1);

end

end

