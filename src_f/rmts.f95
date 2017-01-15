subroutine compute_coeff2nodal(nx, nmtx, mtx)

  implicit none

  !f2py intent(in) nx, nmtx
  !f2py intent(out) mtx
  !f2py depend(nmtx) mtx

  ! Input
  integer, intent(in) :: nx, nmtx

  ! Output
  double precision, intent(out) :: mtx(nmtx, nmtx)

  ! Working
  integer :: iterm1, iterm2, iterm1_list(nx), iterm2_list(nx)
  integer :: nterm, nterm_list(nx)
  double precision :: prod, xval
  logical :: deriv
  integer :: ix, pow

  nterm_list(:) = 4
  nterm = product(nterm_list)

  mtx(:, :) = 0.

  do iterm1 = 1, nterm
     call expandindex(nx, nterm_list, iterm1, iterm1_list)

     do iterm2 = 1, nterm
        call expandindex(nx, nterm_list, iterm2, iterm2_list)

        prod = 1.
        do ix = 1, nx
           if (iterm1_list(ix) .eq. 1) then
              deriv = .False.
              xval = -1.
           else if (iterm1_list(ix) .eq. 2) then
              deriv = .False.
              xval = 1.
           else if (iterm1_list(ix) .eq. 3) then
              deriv = .True.
              xval = -1.
           else if (iterm1_list(ix) .eq. 4) then
              deriv = .True.
              xval = 1.
           else
              print *, 'Error in compute_coeff2nodal'
              call exit(1)
           end if

           pow = iterm2_list(ix) - 1
           if (deriv) then
              if (pow .ge. 1) then
                 prod = prod * pow * xval ** (pow-1)
              else
                 prod = 0.
              end if
           else
              prod = prod * xval ** pow
           end if
        end do

        mtx(iterm1, iterm2) = prod
     end do
  end do

end subroutine compute_coeff2nodal



subroutine compute_uniq2elem(nnz, nx, nelements, data, rows, cols)

  implicit none

  !f2py intent(in) nnz, nx, nelements
  !f2py intent(out) data, rows, cols
  !f2py depend(nx) nelements
  !f2py depend(nnz) data, rows, cols

  ! Input
  integer, intent(in) :: nnz, nx, nelements(nx)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)

  ! Working
  integer :: ielem, iterm, ielem_list(nx), iterm_list(nx)
  integer :: ider, iuniq, ider_list(nx), isid_list(nx), iuniq_list(nx)
  integer :: inz, ix
  integer :: nelem, nterm, nelem_list(nx), nterm_list(nx)
  integer :: ndofs, nuniq, ndofs_list(nx), nuniq_list(nx)
  integer :: der_map(4), sid_map(4)

  der_map(:) = (/ 1, 1, 2, 2 /)
  sid_map(:) = (/ 1, 2, 1, 2 /)

  nelem_list(:) = nelements
  nterm_list(:) = 4
  ndofs_list(:) = 2
  nuniq_list(:) = 1 + nelements

  nelem = product(nelem_list)
  nterm = product(nterm_list)
  ndofs = product(ndofs_list)
  nuniq = product(nuniq_list)

  inz = 0
  do ielem = 1, nelem
     call expandindex(nx, nelem_list, ielem, ielem_list)

     do iterm = 1, nterm
        call expandindex(nx, nterm_list, iterm, iterm_list)

        do ix = 1, nx
           ider_list(ix) = der_map(iterm_list(ix))
           isid_list(ix) = sid_map(iterm_list(ix))
           iuniq_list(ix) = (ielem_list(ix) - 1) + isid_list(ix)
        end do
        call contractindex(nx, ndofs_list, ider_list, ider)
        call contractindex(nx, nuniq_list, iuniq_list, iuniq)

        inz = inz + 1
        data(inz) = 1.0
        rows(inz) = (ielem - 1) * nterm + iterm - 1
        cols(inz) = (ider - 1) * nuniq + iuniq - 1
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error in compute_uniq2elem', inz, nnz
     call exit(1)
  end if

end subroutine compute_uniq2elem



subroutine compute_sec_deriv(kx, njac, nx, nelements, xlimits, jac)

  implicit none

  !f2py intent(in) kx, njac, nx, nelements, xlimits
  !f2py intent(out) jac
  !f2py depend(nx) nelements, xlimits
  !f2py depend(njac) jac

  ! Input
  integer, intent(in) :: kx, njac, nx
  integer, intent(in) :: nelements(nx)
  double precision, intent(in) :: xlimits(nx, 2)

  ! Output
  double precision, intent(out) :: jac(njac, njac)

  ! Working
  integer :: ix
  integer :: igpt, igpt_list(nx)
  integer :: ngpt, ngpt_list(nx)
  integer :: iterm, iterm_list(nx)
  integer :: nterm, nterm_list(nx)
  double precision :: bma_d2(nx), dxb_dx(nx), xval, prod, prod_dx, prod_wts
  integer :: pow
  double precision :: gpts(4), wts(4)

  gpts(1) = -sqrt(3./7. + 2./7. * sqrt(6./5.))
  gpts(2) = -sqrt(3./7. - 2./7. * sqrt(6./5.))
  gpts(3) =  sqrt(3./7. - 2./7. * sqrt(6./5.))
  gpts(4) =  sqrt(3./7. + 2./7. * sqrt(6./5.))
  wts(1) = (18. - sqrt(30.))/36.
  wts(2) = (18. + sqrt(30.))/36.
  wts(3) = (18. + sqrt(30.))/36.
  wts(4) = (18. - sqrt(30.))/36.

  nterm_list(:) = 4
  nterm = product(nterm_list)

  ngpt_list(:) = 4
  ngpt = product(ngpt_list)

  ! x_k = a_k + (b_k - a_k) / n_k * (xb_k + 1) / 2.0
  ! where xb_k is normalized between [-1, 1] (x bar)
  ! and a_k and b_k represent the limits of the integration
  ! and n_k is the number of elements in the kth dimension
  ! dxb_k/dx_k = 1.0 / ((b_k - a_k) / n_k / 2.0)
  bma_d2 = (xlimits(:, 2) - xlimits(:, 1)) / nelements / 2.
  dxb_dx = 1 / bma_d2

  prod_dx = product(bma_d2)

  ! Flattened loop over Gauss points
  do igpt = 1, ngpt
     call expandindex(nx, ngpt_list, igpt, igpt_list)

     ! For each Gauss points, the weight for the n-D integral is
     ! the product of the individual weights in each dimension
     prod_wts = 1.
     do ix = 1, nx
        prod_wts = prod_wts * wts(igpt_list(ix))
     end do

     ! Loop over coefficients
     do iterm = 1, nterm
        call expandindex(nx, nterm_list, iterm, iterm_list)

        ! Loop over input variables in a term in the polynomial
        prod = 1.
        do ix = 1, nx
           xval = gpts(igpt_list(ix))
           pow = iterm_list(ix) - 1
           if (ix .ne. kx) then
              prod = prod * xval ** pow
           else
              if (pow .ge. 2) then
                 ! dxb_dx(ix) ** 2 is because we need the 2nd deriv. w.r.t. x, not xb
                 prod = prod * pow * (pow-1) * xval ** (pow-2) * dxb_dx(ix) ** 2
              else
                 prod = 0.
              end if
           end if
        end do

        jac(igpt, iterm) = prod * sqrt(prod_wts * prod_dx)
     end do
  end do

  ! Normalize by the total volume of the integration domain
  jac = jac / sqrt(product(xlimits(:, 2) - xlimits(:, 1)))

end subroutine compute_sec_deriv



subroutine compute_full_from_block(nnz, nterm, nelem, mat, data, rows, cols)

  implicit none

  !f2py intent(in) nnz, nterm, nelem, mat
  !f2py intent(out) data, rows, cols
  !f2py depend(nnz) data, rows, cols
  !f2py depend(nterm) mat

  ! Input
  integer, intent(in) :: nnz, nterm, nelem
  double precision, intent(in) :: mat(nterm, nterm)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)

  ! Working
  integer :: inz, ielem, i, j, offset

  inz = 0
  do ielem = 1, nelem
     offset = (ielem - 1) * nterm
     do i = 1, nterm
        do j = 1, nterm
           inz = inz + 1
           data(inz) = mat(i, j)
           rows(inz) = offset + i - 1
           cols(inz) = offset + j - 1
        end do
     end do
  end do

end subroutine compute_full_from_block



subroutine compute_jac_sq(kx, nnz, nx, ny, nelems, neval, nrhs, &
     xlimits, xeval, yeval, data, rows, cols, rhs)

  implicit none

  !f2py intent(in) kx, nnz, nx, ny, nelems, neval, nrhs, xlimits, xeval, yeval
  !f2py intent(out) data, rows, cols, rhs
  !f2py depend(nx) nelems, xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(neval, ny) yeval
  !f2py depend(nnz) data, rows, cols
  !f2py depend(nrhs, ny) rhs

  ! Input
  integer, intent(in) :: kx, nnz, nx, ny, nelems(nx), neval, nrhs
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx), yeval(neval, ny)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)
  double precision, intent(out) :: rhs(nrhs, ny)

  ! Working
  integer :: nelem, nterm, nelem_list(nx), nterm_list(nx)
  integer :: inz, ieval, iterm1, iterm2, ix, ielem, irhs
  integer :: ielem_list(nx), iterm1_list(nx), iterm2_list(nx)
  double precision :: xbar(nx)
  double precision, allocatable :: prod(:)
  integer :: pow
  double precision :: bma_d2(nx), dxb_dx(nx)

  nelem_list(:) = nelems
  nterm_list(:) = 4

  nelem = product(nelem_list)
  nterm = product(nterm_list)

  bma_d2 = (xlimits(:, 2) - xlimits(:, 1)) / nelem_list / 2.
  dxb_dx = 1 / bma_d2

  inz = 0
  rhs(:, :) = 0.

  allocate(prod(nterm))

  do ieval = 1, neval
     do ix = 1, nx
        call findinterval(nelem_list(ix), xlimits(ix, :), xeval(ieval, ix), &
             ielem_list(ix), xbar(ix))
     end do
     call contractindex(nx, nelem_list, ielem_list, ielem)

     do iterm1 = 1, nterm ! Flattened loop over terms - rows
        call expandindex(nx, nterm_list, iterm1, iterm1_list)

        prod(iterm1) = 1.
        do ix = 1, nx
           pow = iterm1_list(ix) - 1
           if (ix .ne. kx) then
              prod(iterm1) = prod(iterm1) * xbar(ix) ** pow
           else
              if (pow .ge. 1) then
                 prod(iterm1) = prod(iterm1) * pow * xbar(ix) ** (pow-1) * dxb_dx(ix)
              else
                 prod(iterm1) = 0.
              end if
           end if
        end do
     end do

     do iterm1 = 1, nterm ! Flattened loop over terms - rows
        call expandindex(nx, nterm_list, iterm1, iterm1_list)

        do iterm2 = 1, nterm ! Flattened loop over terms - cols
           call expandindex(nx, nterm_list, iterm2, iterm2_list)

           inz = inz + 1
           data(inz) = prod(iterm1) * prod(iterm2)
           rows(inz) = (ielem - 1) * nterm + iterm1 - 1
           cols(inz) = (ielem - 1) * nterm + iterm2 - 1
        end do

        irhs = (ielem - 1) * nterm + iterm1
        rhs(irhs, :) = rhs(irhs, :) + prod(iterm1) * yeval(ieval, :)
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error: incorrect nnz in compute_jac_sq', inz, nnz
  end if

  deallocate(prod)

end subroutine compute_jac_sq



subroutine compute_jac(ix, jx, nnz, nx, neval, nelements, xlimits, xeval, &
     data, rows, cols)

  implicit none

  !f2py intent(in) ix, jx, nnz, nx, neval, nelements, xlimits, xeval
  !f2py intent(out) data, rows, cols
  !f2py depend(nx) nelements, xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(nnz) data, rows, cols

  ! Input
  integer, intent(in) :: ix, jx, nnz, nx, neval
  integer, intent(in) :: nelements(nx)
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)

  ! Working
  integer :: ieval, inz, kx
  integer :: ielem, iterm, ielem_list(nx), iterm_list(nx)
  integer :: nelem, nterm, nelem_list(nx), nterm_list(nx)
  double precision :: bma_d2(nx), dxb_dx(nx), xbar(nx), prod
  integer :: pow

  nelem_list(:) = nelements
  nterm_list(:) = 4

  nelem = product(nelem_list)
  nterm = product(nterm_list)

  bma_d2 = (xlimits(:, 2) - xlimits(:, 1)) / nelem_list / 2.
  dxb_dx = 1 / bma_d2

  inz = 0
  do ieval = 1, neval
     do kx = 1, nx
        call findinterval(nelem_list(kx), xlimits(kx, :), xeval(ieval, kx), &
             ielem_list(kx), xbar(kx))
     end do
     call contractindex(nx, nelem_list, ielem_list, ielem)

     do iterm = 1, nterm
        call expandindex(nx, nterm_list, iterm, iterm_list)

        prod = 1.
        do kx = 1, nx
           pow = iterm_list(kx) - 1
           if ((kx .ne. ix) .and. (kx .ne. jx)) then
              prod = prod * xbar(kx) ** pow
           else if ((kx .eq. ix) .and. (kx .eq. jx)) then
              if (pow .ge. 2) then
                 prod = prod * pow * (pow-1) * xbar(kx) ** (pow-2) * dxb_dx(kx) * dxb_dx(kx)
              else
                 prod = 0.
              end if
           else
              if (pow .ge. 1) then
                 prod = prod * pow * xbar(kx) ** (pow-1) * dxb_dx(kx)
              else
                 prod = 0.
              end if
           end if
        end do

        inz = inz + 1
        data(inz) = prod
        rows(inz) = ieval - 1
        cols(inz) = (ielem - 1) * nterm + iterm - 1
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error in compute_jac', inz, nnz
     call exit(1)
  end if

end subroutine compute_jac



subroutine compute_ext_dist(nx, neval, ndx, xlimits, xeval, dx)

  implicit none

  !f2py intent(in) nx, neval, ndx, xlimits, xeval
  !f2py intent(out) dx
  !f2py depend(nx) xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(ndx, nx) dx

  ! Input
  integer, intent(in) :: nx, neval, ndx
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx)

  ! Output
  double precision, intent(out) :: dx(ndx, nx)

  ! Working
  integer :: ieval, ix, iterm, nterm, index
  double precision :: work(neval, nx)

  work(:, :) = xeval(:, :)

  nterm = ndx / neval

  ! After these 2 do loops, work is the position vector from the
  ! nearest point in the domain specified by xlimits.
  do ieval = 1, neval
     do ix = 1, nx
        work = xeval(ieval, ix)
        work = max(xlimits(ix, 1), work)
        work = min(xlimits(ix, 2), work)
        work = xeval(ieval, ix) - work
        do iterm = 1, nterm
           index = (ieval - 1) * nterm + iterm
           dx(index, ix) = work(ieval, ix)
        end do
     end do
  end do

end subroutine compute_ext_dist
