% Function: set_data
% Description: Creates graded meshes according to [Def 4.7,Sch98]. Based on
%              my computations, we must have beta>2/3 and sup_T(h_T/rho_T)
%              <=2*(1+sqrt(2))=:kappa.
%              refineRGB is the red refinement, from p1fem.
% Input:
%   - coordinates: Matrix representing the coordinates of vertices.
%   - elements: Matrix representing triangular elements, where each row 
%               contains three vertex indices representing a triangle.
%   - dirichlet: Indices of vertices with Dirichlet boundary conditions.
%   - neumann: Indices of vertices with Neumann boundary conditions.
%   - hmax: Value of the maximum side length of the triangles in the mesh.
% Output:
%   - coordinates: Matrix representing the new coordinates of vertices.
%   - elements: Matrix representing the new elements in the mesh.
%   - dirichlet: Indices of vertices with Dirichlet boundary conditions,
%                the neumann boundary conditions have been replaced by 
%                Dirichlet ones.
%   - beta_previous: Coefficients of the guess of the solution of the 
%                    semilinear problem.

function [coordinates,elements,dirichlet,beta_previous]=...
            set_data(coordinates,elements,dirichlet,neumann,hmax)

    beta=0.40;
    kappa=2*(1+sqrt(2));
    refined_enough = false;
    while ~refined_enough
        nE = size(elements,1);
        % Precompute side lengths for all elements
        x = coordinates(elements(:,[2,3,1]),1)-coordinates(elements,1);
        y = coordinates(elements(:,[2,3,1]),2)-coordinates(elements,2);
        [hT,~] = max(reshape(sqrt(x.^2+y.^2),nE,3),[],2);
        
        % Mark for refinement
        marked = [];
        for i = 1:nE
            coord_vertices = coordinates(elements(i,:),:);
            
            % If T touches (0,0)
            if any(all(coord_vertices == 0,2))
                max_dist = max(vecnorm(coord_vertices'));
                sup_phi_beta = max_dist^beta;
                
                % Check the upper bound of condition (iii)
                if hT(i)/(hmax*sup_phi_beta) > kappa
                    marked(end+1) = i;
                end
            else
                min_dist = shortest_distance_to_origin(coord_vertices);
                inf_phi_beta = min_dist^beta;
                
                % Check the upper bound of condition (ii)
                if hT(i) > kappa*hmax*inf_phi_beta
                    marked(end+1) = i;
                end
            end
        end
        
        % Mesh refinement of the marked elements
        [coordinates,elements,dirichlet,neumann] = refineRGB(coordinates,elements,dirichlet,neumann,marked);
        refined_enough = isempty(marked);
    end 
    %That's the only line that differs from the mixed boundary case
    dirichlet=cat(1,dirichlet,neumann);
    beta_previous=zeros(size(coordinates,1),1);
end
