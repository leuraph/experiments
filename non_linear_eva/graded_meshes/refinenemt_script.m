load elements.dat
load coordinates.dat
load dirichlet.dat
neumann = [];

hmax = 0.0009;

[coordinates,elements,dirichlet,beta_previous] = set_data(coordinates,elements,dirichlet,neumann,hmax);

n_vertices = size(coordinates, 1);
% u = zeros(n_vertices, 1);
% show(elements,coordinates,u);

folder_name = "refined_meshes/hmax-" + hmax + "_n_vertices-" + n_vertices;
mkdir(folder_name);
save(folder_name + "/coordinates.dat", "coordinates", "-ascii","-double");
save(folder_name + "/elements.dat", "elements", "-ascii","-double");
save(folder_name + "/dirichlet.dat", "dirichlet", "-ascii","-double");