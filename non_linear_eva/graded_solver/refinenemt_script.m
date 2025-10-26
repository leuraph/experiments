load elements.dat
load coordinates.dat
load dirichlet.dat
neumann = [];

hmax = 0.002;

[coordinates,elements,dirichlet,beta_previous] = set_data(coordinates,elements,dirichlet,neumann,hmax);

n_vertices = size(coordinates, 1);
% u = zeros(n_vertices, 1);
% show(elements,coordinates,u);

save("refined_meshes/coordinates_refined_n_vertices-" + n_vertices + "_hmax-" + hmax + ".dat", "coordinates", "-ascii","-double");
save("refined_meshes/elements_refined_n_vertices-" + n_vertices + "_hmax-" + hmax + ".dat", "elements", "-ascii","-double");
save("refined_meshes/dirichlet_refined_n_vertices-" + n_vertices + "_hmax-" + hmax + ".dat", "dirichlet", "-ascii","-double");