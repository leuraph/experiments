function show(Elements,Coordinates,u)
%trisurf(Elements,Coordinates(:,1),Coordinates(:,2),u','edgecolor','black','facecolor','interp')
trisurf(Elements,Coordinates(:,1),Coordinates(:,2),u','edgecolor','black','facecolor','white')
view(0,90);
title('Solution of the Problem')