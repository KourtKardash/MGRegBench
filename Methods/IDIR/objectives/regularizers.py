import torch


def compute_hyper_elastic_loss(
    input_coords, output, batch_size=None, alpha_l=1, alpha_a=1, alpha_v=1
):
    """Compute the hyper-elastic regularization loss for 2D case."""
    
    grad_u = compute_jacobian_matrix(input_coords, output, add_identity=False)
    grad_y = compute_jacobian_matrix(input_coords, output, add_identity=True)

    # Compute length loss (membrane energy)
    length_loss = torch.linalg.norm(grad_u, dim=(1, 2))
    length_loss = torch.pow(length_loss, 2)
    length_loss = torch.sum(length_loss)
    length_loss = 0.5 * alpha_l * length_loss

    # Compute area loss (for 2D this is actually the area change penalty)
    # For 2D, the "area" change is represented by the determinant
    det_grad_y = torch.det(grad_y)
    
    # Area loss - penalize deviation from 1 (no area change)
    area_loss = torch.pow(det_grad_y - 1, 2)
    area_loss = torch.sum(area_loss)
    area_loss = alpha_a * area_loss

    # For 2D, volume loss doesn't make sense in the same way as 3D
    # We can use a penalty on negative determinants (folding prevention)
    volume_loss = torch.where(det_grad_y < 0, 
                             torch.pow(-det_grad_y, 2), 
                             torch.zeros_like(det_grad_y))
    volume_loss = torch.sum(volume_loss)
    volume_loss = alpha_v * volume_loss

    # Compute total loss
    loss = length_loss + area_loss + volume_loss

    if batch_size is not None:
        return loss / batch_size
    else:
        return loss / input_coords.shape[0]


def compute_bending_energy(input_coords, output, batch_size=None):
    """Compute the bending energy for 2D case."""
    
    jacobian_matrix = compute_jacobian_matrix(input_coords, output, add_identity=False)

    dx_dx = torch.zeros(input_coords.shape[0], 2, 2)
    dx_dy = torch.zeros(input_coords.shape[0], 2, 2)
    dy_dx = torch.zeros(input_coords.shape[0], 2, 2)
    dy_dy = torch.zeros(input_coords.shape[0], 2, 2)
    
    for i in range(2):
        dx_dx[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 0])
        dx_dy[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 1])
        dy_dx[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 0])
        dy_dy[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 1])

    bending_energy = (
        torch.mean(torch.square(dx_dx[:, :, 0])) +  # d??x/dx?
        torch.mean(torch.square(dx_dx[:, :, 1])) +  # d??x/dy?
        torch.mean(torch.square(dy_dx[:, :, 0])) +  # d??y/dx?  
        torch.mean(torch.square(dy_dx[:, :, 1])) +  # d??y/dy?
        2 * torch.mean(dx_dx[:, :, 0] * dx_dx[:, :, 1]) +  # 2*(d??x/dx?)*(d??x/dy?)
        2 * torch.mean(dy_dx[:, :, 0] * dy_dx[:, :, 1])    # 2*(d??y/dx?)*(d??y/dy?)
    )

    return bending_energy

def compute_jacobian_loss(input_coords, output, batch_size=None):
    """Compute the jacobian regularization loss."""

    # Compute Jacobian matrices
    jac = compute_jacobian_matrix(input_coords, output)

    # Compute determinants and take norm
    loss = torch.det(jac) - 1
    loss = torch.linalg.norm(loss, 1)

    return loss / batch_size


def compute_jacobian_matrix(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output wrt the input for 2D case."""
    
    jacobian_matrix = torch.zeros(input_coords.shape[0], 2, 2)
    for i in range(2):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input for 2D case."""
    
    if grad_outputs is None:
        grad_outputs = torch.ones_like(output)
    
    grad = torch.autograd.grad(
        output, 
        input_coords, 
        grad_outputs=grad_outputs, 
        create_graph=True,
        retain_graph=True
    )[0]
    return grad