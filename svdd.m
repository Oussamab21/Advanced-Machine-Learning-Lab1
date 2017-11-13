clear ;
close all;
randn('seed',1);
pos = [];


n =35;
p = 2;
Xi = randn (n,p) + 2*ones (n ,1) *[0.7 2];



G = Xi*Xi';
nx = diag(G);
e = ones(n,1);

C = 1.5;
 cvx_begin
    cvx_precision best
    variables m(1) cSVDD(2) xi(n)
    dual variables d dp
    minimize(  .5*cSVDD'*cSVDD - m  + C * sum(xi) )
    subject to
       d  :  Xi*cSVDD  >= m + .5*nx - xi;
       dp: xi >= 0;
 cvx_end
 
R = cSVDD'*cSVDD - 2*m;
pos = find(d > eps^.5);


l = 10^-12;
verbose = 1;
[am, lambda, pos] = monqp(2*G,nx,e,1,C,l,verbose);
cm = am'*Xi(pos,:);
Rm = lambda + cm*cm';

% display
figure(3)
set(gcf,'Color',[1,1,1])
visualize_SVDD(Xi,cSVDD,R,pos,'r')
%axis(ax);

% check

[R lambda+am'*G(pos,pos)*am Rm ] % the radius
[ cm' cSVDD  Xi(pos,:)'*am ]   % the centers
%aM = 0*a;
aM(pos) = am;
%[d a aQ aM]

%%%%%%%%%%%%%%%%%%%%%%%%%%%guassian kernal%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kernel = 'gaussian';
kerneloption = 1;
G=svmkernel(Xi,kernel,kerneloption);


lambda = eps^.5;
C = 1.5;
nx = diag(G);
e = ones(n,1);

[am, lambda, pos] = monqp(2*G,nx,e,1,C,l,verbose);
%Rm = lambda + am'*G(pos,pos)*am;

[xtest1 xtest2]  = meshgrid([-1:.01:1]*3+1,[-1:0.01:1]*3+3); 
nn = length(xtest1); 
Xgrid = [reshape(xtest1 ,nn*nn,1) reshape(xtest2 ,nn*nn,1)];


Kgrid = svmkernel(Xgrid,kernel,kerneloption,Xi(pos,:));
ypred = 1  - 2*Kgrid*am - lambda;
ypred = reshape(ypred,nn,nn); 

figure(7)
set(gcf,'Color',[1,1,1])
hold on
contourf(xtest1,xtest2,ypred,50); shading flat; 
[cc,hh]=contour(xtest1,xtest2,ypred,[-1  0],'k','LineWidth',2);
set(gca,'FontSize',14,'FontName','Times','XTick',[ ],'YTick',[ ],'Box','on');  


plot(Xi(:,1),Xi(:,2),'+w','LineWidth',2);
h1 = plot(Xi(pos,1),Xi(pos,2),'ob'); 
set(h1,'LineWidth',1,...
          'MarkerEdgeColor','w',...
          'MarkerSize',15);
hold off


%%%%%%%%%%%%%%%%%%%poly kernal%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C = 6;
%l = 0*eps^.6;

%D = (Xi*Xi');
%N = diag(D);
%D = -2*D + N*ones(1,n) + ones(n,1)*N';
%s = 2.2;
%G =1./(D + s*ones(size(D)));
%nx = diag(G);
%e = ones(n,1);



q = 3;
G = (Xi*Xi' + ones (n)).^q;
C = 1.5;
nx = diag (G);
e = ones (n ,1);


%%%%solver 
[am, lambda, pos] = monqp(2*G,nx,e,1,C,l,verbose);
Rm = lambda + am'*G(pos,pos)*am;





%%%%%%grid and plot 


ss = 4;
[xtest1 xtest2]  = meshgrid([-1:.01:1]*ss+1,[-1:0.01:1]*ss+3); 
nn = length(xtest1); 
Xgrid = [reshape(xtest1 ,nn*nn,1) reshape(xtest2 ,nn*nn,1)]; 

s = 2.2;
N_Kgrid =nx(1);
Dgrid =  (Xgrid*Xi(pos,:)');
normx = sum(Xgrid.^2,2);
normxsup = sum(Xi(pos,:).^2,2);
Dgrid = -2*Dgrid + normx*ones(1,length(pos)) + ones(nn*nn,1)*normxsup' ;  
Kgrid =1./(Dgrid + s*ones(size(Dgrid)));
ypred = N_Kgrid  - 2*Kgrid*am - lambda;
ypred = reshape(ypred,nn,nn); 

figure(8)
set(gcf,'Color',[1,1,1])
hold on
contourf(xtest1,xtest2,(ypred),50); shading flat; 
[cc,hh]=contour(xtest1,xtest2,ypred,[-1  0],'k','LineWidth',2);

plot(Xi(:,1),Xi(:,2),'+w','LineWidth',2);
h1 = plot(Xi(pos,1),Xi(pos,2),'ob'); 
set(h1,'LineWidth',1,...
          'MarkerEdgeColor','w',...
          'MarkerSize',15);
hold off




