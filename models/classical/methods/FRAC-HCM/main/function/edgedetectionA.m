function edge_im=edgedetectionA(A,uA,M,N,p)
heatmatA=zeros(M,N);
sA=A.*double(uA>1);
   for i=2:(M-2)
        for j=2:(N-2)
        maskGray=zeros(1,9) ;       
        for b=1:9 
          [x]=ikomsu(b,i);          
          [y]=jkomsu(b,j);         
          col = sA(x,y).*255;
          maskGray(b) =col;                        
        end           
        [heat_top]=heattrans(maskGray);        
        heatmatA(i,j)=heat_top;
        end
   end 
   for i=2:(M)
        for j=2:(N)
             if(heatmatA(i,j)>p)
             edge_im(i,j)=1;
             else
             edge_im(i,j)=0;            
             end                     
        end
   end  
end

%-----------------------------------------------------------------------------------
%FUNCTIONS
%-----------------------------------------------------------------------------------
function [x] = ikomsu( komsuno, i)

 if(komsuno ==1)
     x=i-1;
 elseif(komsuno ==2)
    x= i;
 elseif(komsuno ==3)
    x= i+1;
 elseif(komsuno ==4)
    x= i-1;
 elseif(komsuno ==5)
    x= i+1;
 elseif(komsuno ==6)
   x= i-1;
 elseif(komsuno ==7)
   x= i;
 elseif(komsuno ==8)
   x= i+1;
 else
    x=i;
 
 end


end

function [y] = jkomsu( komsuno, j )

 if(komsuno ==1)
    y= j-1;
 elseif(komsuno ==2)
    y= j-1;
 elseif(komsuno ==3)
   y= j-1;
 elseif(komsuno ==4)
   y= j;
 elseif(komsuno ==5)
   y= j;
 elseif(komsuno ==6)
   y= j+1;
 elseif(komsuno ==7)
   y= j+1;
 elseif(komsuno ==8)
   y= j+1;
 else
    y= j;
 
 end


end

