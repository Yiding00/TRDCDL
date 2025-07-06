clear
clc

% 读取
load AAL90Class
load AAL90_8Region
load positive_matrix.mat
load negative_matrix.mat

enhanced = negative_matrix>0.02;
weakened = positive_matrix>0.02;

% 生成200x1随机分类编号
Class=class;
className=region;
% CC=circosChart(Data,Class);
figure
subplot(1,2,1)
CC=DirectCircosDiagram(weakened,Class,'ClassName',className);
CC=CC.draw();
CC.setClassLabel('Color',[0,0,0],'FontName','Times new Roman','FontSize',14)
title('Weakened')

subplot(1,2,2)
CC1=DirectCircosDiagram(enhanced,Class,'ClassName',className);
CC1=CC1.draw();
CC1.setClassLabel('Color',[0,0,0],'FontName','Times new Roman','FontSize',14)
title('Enhanced')

