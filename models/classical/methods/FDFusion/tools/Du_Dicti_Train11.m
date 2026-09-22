function [Xh, Dhs,Dhl,Dls,Dll,Zls,Zll,H] = Du_Dicti_Train11( Xh,Xhs,Xhl,Dhs,Dhl,Xl,Xls,Xll,Dls,Dll )
lamda1=1;
lamda2=0.001;
lamda3=1.5;
lamda4=0.01;
lamda5=0.00001;
H = eye(size(Dhl, 2));

Zhs=IniZ(Xhs,Dhs,0.01);
Zls=IniZ(Xls,Dls,0.01);
Zhl=IniZ(Xhl,Dhl,0.01);
Zll=IniZ(Xll,Dll,0.01);


[a,b]=size(Zls);
A=ones(a,a);
for iter=1:10
    iter
    tic
Xhs=(1/2)*(Xh-Xhl+Dhs*Zhs);
%************Xhl*************
Xhll=[Xh-Xhs;Dhl*Zhl];
[m,n]=size(Xh);
[J1]=singular_value_shrinkage(Xhll,lamda3); 
Xhl= J1(1:m,:); 
% Xhl=(1/2)*(Xh-Xhs+Dhl*Zhl);
% [P1]=singular_value_shrinkage(Xhl,lamda3/mu);
% Xhl=(1/3)*(Xh-Xhs+Dhl*Zhl+P1);
%************Zhs*************
Xhss=[Xhs;sqrt(lamda1)*Dhs*H*Zls];
% Xhss=[Xhs;H*Zls];
% EE=eye(size(Dhs,2));
Dhss=[Dhs;sqrt(lamda1)*Dhs];
% Dhss=[Dhs;EE];
%********TwIST**********
Zhs = UpadteZhs( Xhss,Dhss,lamda4 );
%************Zhl*************
E1=eye(size(Dhl,2));
 Zhl=(Dhl'*Dhl+(lamda2/a^2)*(A'*A)+0.00001*E1)^(-1)*(Dhl'*Xhl+(lamda2/a)*A'*H*Zll);

%************Xls*************
Xls=(1/2)*(Xl-Xll+Dls*Zls);
%************Xll*************

Xlll=[Xl-Xls;Dll*Zll];
[p,q]=size(Xls);
[Q1]=singular_value_shrinkage(Xlll,lamda3); 
Xll= Q1(1:p,:); 
%************Zls*************

Xlss=[Xls;sqrt(lamda1)*Dhs*Zhs];
% Xlss=[Xls;Zhs];
Dlss=[Dls;sqrt(lamda1)*Dhs*H];
[ Zls ] = UpdateZls( Xlss,Dlss,lamda4);
%************Zll*************
E2=eye(size(Dll,2));
Zll=(Dll'*Dll+lamda2*(H'*H)+E2*0.000001)^(-1)*(Dll'*Xll+(lamda2/a)*H'*A*Zhl);
 
 %************H*************
 B=(Dhs'*Dhs);
 C=lamda2*(Zll*Zll')/(Zls*Zls')+lamda5*(Zls*Zls')^(-1);
% C=lamda2*(Zll*Zll')/(Zls*Zls');
 D=(Dhs'*Dhs*Zhs*Zls')/(Zls*Zls')+(lamda2/a)*(A*Zhl*Zll')/(Zls*Zls');
 H=sylvester(B,C,D);
%************Dhs*************
ZZ=zeros(size(Xhs));
AA=[Xhs ZZ];
BB=[Zhs sqrt(lamda1)*(Zhs-H*Zls)];
Dhs=learn_basis(AA, BB, 10);
%************Dhl*************
Dhl=learn_basis(Xhl, Zhl, 1);
%************Dls*************
Dls=learn_basis(Xls, Zls, 1);
%************Dll*************
Dll=learn_basis(Xll, Zll, 1);

toc
end
