import numpy as np 

class Truss_3D:
    
    def __init__(self, nodes, elements, supports, forces, elasticity, cross_area):
        '''
        Initializes truss class object. User should be aware of its units for consistency of solution.
        
        Parameters
        ----------
        
        nodes: dictionary 
               nodal name/mark followed by the [x,y] coordinate in an array form.
        
        elements: dictionary
                  member or element name followed by the connecting nodes in a form of an array.
                  
        supports: dictionary
                  applies supports at the node name/mark followed by the an array support condition as follows:
                  if array == [1,1]: Pin support
                  if array == [0,1]: Roller support free to move abot the x-axis
                  if array == [1,0]: Roller support free to move abot the y-axis
                  
        forces: dictionary
                applies forces at the node name/mark, followed by an array of [x,y] coordinate. Positive values for x-axis indicate right direction. Positive values for y-axis indicate up direction
        
        elasticity: dictionary
                    member's modulus of elasticity. Member's name/mark followed by its modulus of elasticity
                    
        cross_area: dictionary
                    member's cross-sectional area. Member's name/mark followed by its cross-sectional area
                  
        '''
        
        
        self.nodes = nodes
        self.elements = elements
        self.supports = supports
        self.forces = forces
        self.elasticity = elasticity
        self.cross_area = cross_area
        self.K_global_ = []
        self.displacements_ = [] 
        self.reactions_ = []
        self.member_forces_ = [] 
        self.member_stresses_ = []


    def __Direction_Cosines(self, element, nodes, elements):
        from_node = elements[element][0] 
        to_node = elements[element][1]
        from_point = nodes[from_node]
        to_point= nodes[to_node]

        x1 = from_point[0]
        y1 = from_point[1]
        z1 = from_point[2]

        x2 = to_point[0]
        y2 = to_point[1]
        z2 = to_point[2]

        length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

        cx = (x2-x1)/length
        cy = (y2-y1)/length
        cz = (z2-z1)/length
    
        return (cx, cy, cz, length)


    def __Assemble_Stiffness_Matrix(self, element, elasticity, area, nodes, elements):
        cx, cy, cz, L = self.__Direction_Cosines(element, nodes, elements)
        k = elasticity[element] * area[element] / L * np.array([[1, -1],
                                                                [-1, 1]])
        T = np.array([[cx,cy,cz,0,0,0],
                    [0,0,0,cx,cy,cz]])
        K = np.transpose(T).dot(k).dot(T)
        return K


    def __Assemble_Global_Stiffness(self, K, k, i, j):
        dofs = [3*i-3,3*i-2, 3*i-1, 3*j-3, 3*j-2, 3*j-1]
        K[np.ix_(dofs,dofs)] += k
        return K


    def __Support_Vector(self, restrained_dofs, nodes):
        dofs = np.zeros([3 * len(nodes)])

        for dof in restrained_dofs:
            dofs[dof * 3 - 3] = restrained_dofs[dof][0]
            dofs[dof * 3 - 2] = restrained_dofs[dof][1]
            dofs[dof * 3 - 1] = restrained_dofs[dof][1]

        Support_Vector = []

        for i, dof in enumerate(dofs):
            if dof == 1:
                Support_Vector.append(i + 1)
        
        return Support_Vector


    def __Apply_Boundary_Conditions(self, restrained_dofs, K_global):
        dofs = []

        for i,j in enumerate(restrained_dofs):
            dofs.append(j - 1)

        k = np.delete(K_global, obj = dofs, axis = 0)
        k_new = np.delete(k, obj = dofs, axis = 1)
        return k_new


    def __Assemble_Force_Vector(self, forces, restrained_dofs, nodes):
        # Create force vector
        f = np.zeros([3 * len(nodes)])

        # extracts forces along x and y
        for force in forces:
            f[force * 3 - 3] = forces[force][0]
            f[force * 3 - 2] = forces[force][1]
            f[force * 3 - 1] = forces[force][2]

        # create dof list
        dofs = []

        # loops and appends restrained dofs and appends it to the list dof
        for i, j in enumerate(restrained_dofs):
            dofs.append(j-1)

        # removes force vector unecessary rows
        f_new = np.delete(f, obj = dofs, axis = 0)
        return f_new


    def __Truss_Global_Displacement(self, displacements, Support_Vector, nodes):
        # Create New Support Vector in python indexing
        support_vector_new = [x - 1 for x in Support_Vector]

        # Create Displacement Vector
        displacement_vector = np.zeros(3*len(nodes))

        # Creating global displacement vector that looks for all elements in the support vector and replacing the with value of 0 of that particular index
        j = 0

        # Looping displacement vectors indexes and looks for any of the value within the array "Support_Vector". Replaces with 0 if the value of the index is equal to any of the support vector.
        for i, _ in enumerate(displacement_vector):
            if np.any(np.in1d(support_vector_new, i)):
                displacement_vector[i] = 0
            else:
                displacement_vector[i] = displacements[j]
                j += 1

        return displacement_vector


    def __Solve_Reactions(self, K_global, displacement_vector):
        return np.round(K_global.dot(displacement_vector))


    def __Element_Displacement(self, element_number, global_displacement, elements):

        fromNode = elements[element_number][0]
        toNode = elements[element_number][1]

        u = [3 * fromNode - 2, 3 * fromNode - 1, 3 * fromNode, 
            3 * toNode - 2, 3 * toNode - 1, 3 * toNode]

        elem_displacements = []

        for _, u_node in enumerate(u):
            elem_displacements.append(global_displacement[u_node - 1])

        return np.round(elem_displacements,5)


    def __Solve_Member_Force(self, element, displacement_vector, area, elasticity, nodes, elements):
        cx, cy, cz, L = self.__Direction_Cosines(element, nodes, elements)
        T = np.array([[cx,cy,cz,0,0,0],
                    [0,0,0,cx,cy,cz]])
        u = T.dot(displacement_vector[element-1])
        k = elasticity[element] * area[element] / L * np.array([[1, -1],
                                                                [-1, 1]])
        member_force = k.dot(u)
        return member_force # 1st vector: positive = compression, negative = Tension


    def __Solve_Member_Stress(self, element, displacement_vector, elasticity, nodes, elements):
        cx, cy, cz, L = self.__Direction_Cosines(element, nodes, elements)
        T = np.array([[cx,cy,cz,0,0,0],
                    [0,0,0,cx,cy,cz]])
        u = T.dot(displacement_vector[element-1])
        k = elasticity[element] / L * np.array([[1, -1],
                                                [-1, 1]])
        member_stress = k.dot(u)
        return member_stress # 1st vector: positive = compression, negative = Tension


    def Solve(self):
        '''
        Solves the 3D Truss.
        
        Output Parameters
        -----------------
        
        displacements_: dictionary
                        global displacement of each node. name of each node accompanied by its displacement values along x and y respectively 
        
        reactions_: dictionary
                    global reactions of the truss. name of each node accompanied by its force values along x and y respectively 
        
        member_forces_: dictionary
                        member forces of the truss. name of each member accompanied by its force values. positive (+) values are in tension and negative (-) values are in compression
        
        member_stresses_: dictionary
                          member stresses of the truss. name of each member accompanied by its stress values. positive (+) values are in tension and negative (-) values are in compression
                          
        K_global: numpy array
                  returns the Global Stiffness Matrix of the Truss
        
        '''       
        
        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        forces = self.forces
        elasticity = self.elasticity
        cross_area = self.cross_area
        
        # Step 1: Get all Member Lengths
        member_lengths = []

        for element in elements:
            _, _, _, L = self.__Direction_Cosines(element, nodes, elements)
            member_lengths.append(L)

        # Step 2: Assemble Stiffness Matrix for All members
        k_elems = []

        for element in elements:
            k_elems.append(self.__Assemble_Stiffness_Matrix(element, elasticity, cross_area, nodes, elements))

        # Step 3: Assemble Global Stiffness Matrix
        K_global = np.zeros([3*len(nodes), 3*len(nodes)])

        for i, _ in enumerate(k_elems):
            K_global = self.__Assemble_Global_Stiffness(K_global, k_elems[i], elements[i+1][0], elements[i+1][1])

        # Step 4: Apply Boundary conditions to reduce the Global Stiffness Matrix 
        Support_Vector = self.__Support_Vector(supports, nodes)
        K_new = self.__Apply_Boundary_Conditions(Support_Vector, K_global)

        # Step 5: Reduce Force Vector
        f_new = self.__Assemble_Force_Vector(forces, Support_Vector, nodes)

        # Step 6: Solve for Displacement
        displacements = np.linalg.inv(K_new).dot(f_new.transpose())

        # Step 7: Create Global Displacement Vector
        global_displacements = self.__Truss_Global_Displacement(displacements, Support_Vector, nodes)

        # Step 8: Solve for Reactions 
        reactions = self.__Solve_Reactions(K_global, global_displacements)

        # Step 9: Solve Member Displacements
        element_displacements = []

        for element in elements:
            element_displacements.append(self.__Element_Displacement(element, global_displacements, elements))

        # Step 10: Solve Member Forces
        member_forces = []

        for element in elements:
            member_forces.append(self.__Solve_Member_Force(element, element_displacements, cross_area, elasticity, nodes, elements))
        member_forces = {key: member_forces[key-1][1] for key in elements}

        # Step 11: Solve Member Stresses
        member_stresses = []

        for element in elements:
            member_stresses.append(self.__Solve_Member_Stress(element, element_displacements, elasticity, nodes, elements))
        member_stresses = {key: member_stresses[key-1][1] for key in elements}

        # Variable lists

        self.displacements_ = global_displacements
        self.reactions_ = reactions
        self.member_forces_ = member_forces
        self.member_stresses_ = member_stresses
        self.K_global_ = K_global

        lengths = {}
        for key, length in enumerate(L):
            lengths.update({key+1: length})
        self.member_lengths_ = lengths
