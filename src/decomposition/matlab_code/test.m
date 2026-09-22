% pfilter = '9/7';
% % pfilter = '5/3';
% % pfilter = 'Burt';
% % pfilter = 'pkva';
% % pfilter = 'db1';



% [h, g] = pfilters(pfilter);
% disp('h:');
% disp(h);
% disp('g:');
% disp(g);

A = [
    80, 20, 85, 90;
    75, 25, 78, 82;
    80, 22, 88, 92;
    78, 24, 80, 85
];

[cA1, cH1, cV1, cD1] = swt2(A, 1, 'db1');
disp('cA1:');
disp(cA1);
[cA2, cH2, cV2, cD2] = dwt2(A, 'db1');
disp('cA2:');
disp(cA2);