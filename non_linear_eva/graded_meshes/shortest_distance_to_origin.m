function min_distance = shortest_distance_to_origin(vertices)    
    % Calculate distances from vertices to the origin (0,0)
    distances = sqrt(vertices(:, 1).^2 + vertices(:, 2).^2);
    
    % Initialize min_distance with the minimum distance from the vertices
    min_distance = min(distances);
    
    % Loop through each edge of the triangle
    for i = 1:3
        % Get the endpoints of the edge
        p1 = vertices(i, :);
        p2 = vertices(mod(i, 3) + 1, :);
        
        % Calculate the perpendicular distance from the origin to the edge
        edge_vector = p2 - p1;
        point_vector = -p1;
        
        % Calculate the projection of point_vector onto edge_vector
        t = dot(point_vector, edge_vector) / dot(edge_vector, edge_vector);
        
        % Ensure t is between 0 and 1 to stay within the edge segment
        t = max(0, min(1, t));
        
        % Calculate the closest point on the edge to the origin
        closest_point = p1 + t * edge_vector;
        
        % Calculate the distance from the origin to the closest point on the edge
        distance_to_edge = norm(closest_point);
        
        % Update min_distance if a smaller distance is found
        min_distance = min(min_distance, distance_to_edge);
    end
end
