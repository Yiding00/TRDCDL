classdef DirectCircosDiagram
% @author : Yiding00
% highly adopted from https://www.mathworks.com/matlabcentral/fileexchange/118655-circos-plot

    properties
        ax,arginList={'ColorOrder','ClassName','PartName'}
        ColorOrder=[134,184,176;131,147,149;201,179,80;
            198,132,111;170,151,173;93,150,172;190,123,181;117,116,192;135,187,171;135,148,142;200,175,82;
            191,136,115;171,152,175;95,158,179;195,120,187;110,115,196]./255;
        ClassName,PartName
        Data,Class,indexInClass,colorSet={[]}
        classSet,classNum,classSize,classRatio,classTheta
        lineHdl,partLabelHdl,classLabelHdl,scatterHdl
    end

    methods
        function obj=DirectCircosDiagram(Data,Class,varargin)
            obj.Data=Data;
            obj.Class=Class(:);
            obj.classSet=unique(Class);
            obj.classNum=length(obj.classSet);

            obj.indexInClass=zeros(length(obj.Class),1);
            % 计算比例
            for i=1:obj.classNum
                tClassBool=obj.classSet(i)==obj.Class;
                tCumsumBool=cumsum(tClassBool);
                obj.classSize(i)=sum(tClassBool);
                obj.indexInClass(tClassBool)=tCumsumBool(tClassBool);
            end
            obj.classRatio=obj.classSize./sum(obj.classSize);
            % disp(char([64 97 117 116 104 111 114 32 58 32 115 108 97 110 100 97 114 101 114]))
            obj.ColorOrder=[obj.ColorOrder;rand([obj.classNum,3])];
            for i=1:size(obj.Data,1)
                obj.PartName{i}='';
            end
            for i=1:obj.classNum
                obj.ClassName{i}=['Class ',num2str(i)];
            end
            % 获取其他数据
            for i=1:2:(length(varargin)-1)
                tid=ismember(obj.arginList,varargin{i});
                if any(tid)
                obj.(obj.arginList{tid})=varargin{i+1};
                end
            end
        end

        function obj=draw(obj)
            obj.ax=gca;hold on
            obj.ax.XLim=[-1.2,1.2];
            obj.ax.YLim=[-1.2,1.2];
            obj.ax.XTick=[];
            obj.ax.YTick=[];
            obj.ax.XColor='none';
            obj.ax.YColor='none';
            obj.ax.PlotBoxAspectRatio=[1,1,1];

            % 调整初始界面大小
            fig=obj.ax.Parent;
            % fig.Color = 'k';
            % obj.ax.Color = 'k';
            if max(fig.Position(3:4))<600
                fig.Position(3:4)=1.8.*fig.Position(3:4);
                fig.Position(1:2)=fig.Position(1:2)./3;
            end

            sepTheta=1/length(obj.Class);
            cumTheta=[0,cumsum(obj.classRatio)];

            % 计算每一类中每一个元素的角度
            for i=1:obj.classNum
                obj.classTheta(i).T=linspace(cumTheta(i),cumTheta(i+1),obj.classSize(i)+1).*2.*pi;
            end

            if isempty(obj.PartName{1})&&isempty(obj.PartName{2})
                tdis=1.12;
            else
                tdis=1.22;
            end

            outerRadius = 1.05; % 环的外半径
            innerRadius = 1.0; % 环的内半径
            
            % 绘制每个类别的环状区域
            for i = 1:obj.classNum

                % 环的角度范围
                theta=linspace(cumTheta(i),cumTheta(i+1),obj.classSize(i)+1).*2.*pi;
            
                % 计算外圆和内圆的坐标
                x_outer = outerRadius * cos(theta);
                y_outer = outerRadius * sin(theta);
                x_inner = innerRadius * cos(fliplr(theta)); % 内环角度反转以完成环的封闭
                y_inner = innerRadius * sin(fliplr(theta));
                
                % 合并外圆和内圆的坐标
                x_ring = [x_outer, x_inner];
                y_ring = [y_outer, y_inner];
            
                % 绘制填充环状区域
                fill(x_ring, y_ring, obj.ColorOrder(i, :), 'EdgeColor', 'none');
            
                % 绘制类别标签
                CTi = mean(obj.classTheta(i).T);
                rotation = CTi / pi * 180;
                if rotation > 0 && rotation < 180
                    obj.classLabelHdl(i) = text(cos(CTi) * tdis, sin(CTi) * tdis, obj.ClassName{i}, 'FontSize', 14/90*length(obj.Class), 'FontName', 'Arial', ...
                        'HorizontalAlignment', 'center', 'Rotation', -(.5 * pi - CTi) / pi * 180);
                else
                    obj.classLabelHdl(i) = text(cos(CTi) * tdis, sin(CTi) * tdis, obj.ClassName{i}, 'FontSize', 14/90*length(obj.Class), ...
                        'HorizontalAlignment', 'center', 'Rotation', -(1.5 * pi - CTi) / pi * 180);
                end
            end
            % 绘制文字
            for i=1:size(obj.Data,1)

                Ci=obj.Class(i);Pi=obj.indexInClass(i);
                Ti=obj.classTheta(Ci).T(Pi);
                rotation=Ti/pi*180;
                 
                if rotation>90&&rotation<270
                    rotation=rotation+180;
                    obj.partLabelHdl(i)=text(cos(Ti).*1.03,sin(Ti).*1.03,obj.PartName{i},'Rotation',rotation,'HorizontalAlignment','right','FontSize',8/90*length(obj.Class));
                else
                    obj.partLabelHdl(i)=text(cos(Ti).*1.03,sin(Ti).*1.03,obj.PartName{i},'Rotation',rotation,'FontSize',8/90*length(obj.Class));
                end
            end

            % 计算类与类之间的渐变色
            t2=linspace(0,1,200);t1=1-t2;
            for i=1:obj.classNum
                for j=1:obj.classNum
                    C1=obj.ColorOrder(i,:);
                    C2=obj.ColorOrder(j,:);
                    obj.colorSet{i,j}=uint8([t1.*C1(1)+t2.*C2(1);
                        t1.*C1(2)+t2.*C2(2);
                        t1.*C1(3)+t2.*C2(3)
                        ones(1,200).*.6].*255);
                end
            end
            % 画线并赋予颜色
            obj.colorSet
            size(obj.Data,1)
            for i=1:size(obj.Data,1)
                for j=1:size(obj.Data,1)
                    if obj.Data(i,j)>0 && i~=j
                        Ci=obj.Class(i);Pi=obj.indexInClass(i);
                        Cj=obj.Class(j);Pj=obj.indexInClass(j);
                        Ti=obj.classTheta(Ci).T(Pi)+2*pi*sepTheta/2;
                        Tj=obj.classTheta(Cj).T(Pj)+2*pi*sepTheta/2;
                        Xij=[cos(Ti),0,cos(Tj)]';
                        Yij=[sin(Ti),0,sin(Tj)]';
                        XYb=bezierCurve([Xij,Yij],200);
                        % 绘制连接线
                        obj.lineHdl(i,j)=plot(XYb(:,1),XYb(:,2),'-','LineWidth',1);
                        % 绘制箭头
                        % 计算箭头的方向向量（从倒数第二个点指向最后一个点）
                        arrowDirX = XYb(end, 1) - XYb(end-1, 1);
                        arrowDirY = XYb(end, 2) - XYb(end-1, 2);
                        
                        % 归一化方向向量
                        arrowDirLength = sqrt(arrowDirX^2 + arrowDirY^2);
                        arrowDirX = arrowDirX / arrowDirLength;
                        arrowDirY = arrowDirY / arrowDirLength;
                        
                        % 设置箭头长度和角度
                        arrowLength = 0.05;  % 箭头的长度
                        arrowAngle = pi/8;   % 箭头两边的角度（30度）
                        
                        % 计算箭头两边的方向向量
                        leftArrowX = cos(arrowAngle) * arrowDirX - sin(arrowAngle) * arrowDirY;
                        leftArrowY = sin(arrowAngle) * arrowDirX + cos(arrowAngle) * arrowDirY;
                        rightArrowX = cos(-arrowAngle) * arrowDirX - sin(-arrowAngle) * arrowDirY;
                        rightArrowY = sin(-arrowAngle) * arrowDirX + cos(-arrowAngle) * arrowDirY;
                        
                        % 缩放向量，使其长度等于 arrowLength
                        leftArrowX = leftArrowX * arrowLength;
                        leftArrowY = leftArrowY * arrowLength;
                        rightArrowX = rightArrowX * arrowLength;
                        rightArrowY = rightArrowY * arrowLength;
                        
                        % % 在贝塞尔曲线的末端绘制箭头两边的线，确保颜色与贝塞尔曲线颜色匹配
                        % plot([XYb(end, 1), XYb(end, 1) - leftArrowX], [XYb(end, 2), XYb(end, 2) - leftArrowY], ...
                        %     'Color', obj.ColorOrder(Cj, :), 'LineWidth', 1);
                        % plot([XYb(end, 1), XYb(end, 1) - rightArrowX], [XYb(end, 2), XYb(end, 2) - rightArrowY], ...
                        %     'Color', obj.ColorOrder(Cj, :), 'LineWidth', 1);
                                    % 计算三角形顶点
                        arrowTipX = XYb(end, 1);
                        arrowTipY = XYb(end, 2);
                        leftCornerX = arrowTipX - leftArrowX;
                        leftCornerY = arrowTipY - leftArrowY;
                        rightCornerX = arrowTipX - rightArrowX;
                        rightCornerY = arrowTipY - rightArrowY;
                                    
                        % 绘制实心三角形箭头
                        fill([arrowTipX, leftCornerX, rightCornerX], [arrowTipY, leftCornerY, rightCornerY], obj.ColorOrder(Cj, :), 'EdgeColor', 'none');


                    end
                end
            end
            pause(1e-16)
            for i=1:size(obj.Data,1)
                for j=1:size(obj.Data,1)
                    if obj.Data(i,j)>0 && i~=j
                        Ci=obj.Class(i);
                        Cj=obj.Class(j);
                        set(get(obj.lineHdl(i,j),'Edge'),'ColorBinding','interpolated','ColorData',obj.colorSet{Ci,Cj})

                    end
                end
            end
            % 贝塞尔函数
            function pnts=bezierCurve(pnts,N)
                t=linspace(0,1,N);
                p=size(pnts,1)-1;
                coe1=factorial(p)./factorial(0:p)./factorial(p:-1:0);
                coe2=((t).^((0:p)')).*((1-t).^((p:-1:0)'));
                pnts=(pnts'*(coe1'.*coe2))';
            end
        end
        % 设置线除了颜色的其他属性
        function setLine(obj,varargin)
             for i=1:size(obj.Data,1)
                for j=1:size(obj.Data,1)
                    if obj.Data(i,j)>0 && i~=j
                        set(obj.lineHdl(i,j),varargin{:})
                    end
                end
             end
        end
        % 设置线颜色
        function setColor(obj,N,color)
            obj.ColorOrder(N,:)=color;
            t2=linspace(0,1,200);t1=1-t2;
            for i=1:obj.classNum
                set(obj.scatterHdl(i),'CData',obj.ColorOrder(i,:))
                for j=1:obj.classNum
                    C1=obj.ColorOrder(i,:);
                    C2=obj.ColorOrder(j,:);
                    obj.colorSet{i,j}=uint8([t1.*C1(1)+t2.*C2(1);
                        t1.*C1(2)+t2.*C2(2);
                        t1.*C1(3)+t2.*C2(3)
                        ones(1,200).*.6].*255);
                end
            end
            for i=1:size(obj.Data,1)
                for j=1:size(obj.Data,1)
                    if obj.Data(i,j)>0 && i~=j
                        Ci=obj.Class(i);
                        Cj=obj.Class(j);
                        set(get(obj.lineHdl(i,j),'Edge'),'ColorBinding','interpolated','ColorData',obj.colorSet{Ci,Cj})
                    end
                end
            end
        end

        % 设置标签
        function setPartLabel(obj,varargin)
            for i=1:size(obj.Data,1)
                set(obj.partLabelHdl(i),varargin{:});
            end
        end

        function setClassLabel(obj,varargin)
            for i=1:obj.classNum
                set(obj.classLabelHdl(i),varargin{:});
            end
        end
    end

end