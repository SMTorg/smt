subroutine tpsblockd2(kx, order, njac, nx, nelements, wb, xlimits, jac)

  implicit none

  !f2py intent(in) kx, order, njac, nx, nelements, wb, xlimits
  !f2py intent(out) jac
  !f2py depend(nx) nelements, xlimits
  !f2py depend(njac) jac

  ! Input
  integer, intent(in) :: kx, order, njac, nx
  integer, intent(in) :: nelements(nx)
  double precision, intent(in) :: wb, xlimits(nx, 2)

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
  double precision :: gpts(order), wts(order)

  if (order .eq. 3) then
     gpts(1) = -sqrt(3./5.)
     gpts(2) =  0.0
     gpts(3) =  sqrt(3./5.)
     wts(1) = 5./9.
     wts(2) = 8./9.
     wts(3) = 5./9.
  else if (order .eq. 4) then
     gpts(1) = -sqrt(3./7. + 2./7. * sqrt(6./5.))
     gpts(2) = -sqrt(3./7. - 2./7. * sqrt(6./5.))
     gpts(3) =  sqrt(3./7. - 2./7. * sqrt(6./5.))
     gpts(4) =  sqrt(3./7. + 2./7. * sqrt(6./5.))
     wts(1) = (18. - sqrt(30.))/36.
     wts(2) = (18. + sqrt(30.))/36.
     wts(3) = (18. + sqrt(30.))/36.
     wts(4) = (18. - sqrt(30.))/36.
  else
     print *, 'Error in tpsgaussd2', order
     call exit(1)
  end if     

  nterm_list(:) = order
  nterm = product(nterm_list)
  
  ngpt_list(:) = order
  ngpt = product(ngpt_list)

  bma_d2 = (xlimits(:, 2) - xlimits(:, 1)) / nelements / 2.
  dxb_dx = 1 / bma_d2

  prod_dx = product(bma_d2)

  do igpt = 1, ngpt ! Flattened loop over Gauss points
     call expandindex(nx, ngpt_list, igpt, igpt_list)
     
     prod_wts = 1.
     do ix = 1, nx
        prod_wts = prod_wts * wts(igpt_list(ix))
     end do
     
     do iterm = 1, nterm
        call expandindex(nx, nterm_list, iterm, iterm_list)

        prod = 1.
        do ix = 1, nx
           xval = gpts(igpt_list(ix))
           pow = iterm_list(ix) - 1
           if (ix .ne. kx) then
              prod = prod * xval ** pow
           else
              if (pow .ge. 2) then
                 prod = prod * pow * (pow-1) * xval ** (pow-2) * dxb_dx(ix) * dxb_dx(ix)
              else
                 prod = 0.
              end if
           end if              
        end do

        jac(igpt, iterm) = prod * sqrt(prod_wts * prod_dx * wb)
     end do
  end do
  
end subroutine tpsblockd2



subroutine tpsblockdiag(nnz, nterm, nelem, mat, data, rows, cols)

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

end subroutine tpsblockdiag



subroutine tpsjac(ider, kx, order, nnz, nx, neval, nelements, xlimits, xeval, &
     data, rows, cols)

  implicit none

  !f2py intent(in) ider, kx, order, nnz, nx, neval, nelements, xlimits, xeval
  !f2py intent(out) data, rows, cols
  !f2py depend(nx) nelements, xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(nnz) data, rows, cols

  ! Input
  integer, intent(in) :: ider, kx, order, nnz, nx, neval
  integer, intent(in) :: nelements(nx)
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)

  ! Working
  integer :: ieval, inz, ix
  integer :: ielem, iterm, ielem_list(nx), iterm_list(nx)
  integer :: nelem, nterm, nelem_list(nx), nterm_list(nx)
  double precision :: bma_d2(nx), dxb_dx(nx), xbar(nx), prod
  integer :: pow

  nelem_list(:) = nelements
  nterm_list(:) = order

  nelem = product(nelem_list)
  nterm = product(nterm_list)

  bma_d2 = (xlimits(:, 2) - xlimits(:, 1)) / nelem_list / 2.
  dxb_dx = 1 / bma_d2

  inz = 0
  do ieval = 1, neval
     do ix = 1, nx
        call findinterval(nelem_list(ix), xlimits(ix, :), xeval(ieval, ix), &
             ielem_list(ix), xbar(ix))
     end do
     call contractindex(nx, nelem_list, ielem_list, ielem)

     do iterm = 1, nterm
        call expandindex(nx, nterm_list, iterm, iterm_list)

        prod = 1.
        do ix = 1, nx
           pow = iterm_list(ix) - 1
           if (ix .ne. kx) then
              prod = prod * xbar(ix) ** pow
           else if (ider .eq. 1) then
              if (pow .ge. 1) then
                 prod = prod * pow * xbar(ix) ** (pow-1) * dxb_dx(ix)
              else
                 prod = 0.
              end if
           else if (ider .eq. 2) then
              if (pow .ge. 2) then
                 prod = prod * pow * (pow-1) * xbar(ix) ** (pow-2) * dxb_dx(ix) * dxb_dx(ix)
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
     print *, 'Error in tpsjac', inz, nnz
     call exit(1)
  end if

end subroutine tpsjac



subroutine tpscontjac(nnz, nnz2, nx, nelems, &
     data, rows, cols, rows2, cols2)

  implicit none

  !f2py intent(in) nnz, nnz2, nx, nelems
  !f2py intent(out) data, rows, cols, rows2, cols2
  !f2py depend(nx) nelems
  !f2py depend(nnz) data, rows, cols
  !f2py depend(nnz2) rows2, cols2

  ! Input
  integer, intent(in) :: nnz, nnz2, nx, nelems(nx)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)
  integer, intent(out) :: rows2(nnz2), cols2(nnz2)

  ! Working
  integer :: nelem, nterm, nuniq
  integer :: nelem_list(nx), nterm_list(nx), nuniq_list(nx)
  integer :: inz, inz2, ipt, iterm, ielem, ix, iuniq
  integer :: ipt_list(nx), iterm_list(nx)
  integer :: ielem_list(nx), iuniq_list(nx)
  integer :: pow
  double precision :: xpts(3), xval, prod

  nelem_list(:) = nelems
  nterm_list(:) = 3
  nuniq_list(:) = 1 + 2 * nelem_list

  nelem = product(nelem_list)
  nterm = product(nterm_list)
  nuniq = product(nuniq_list)

  xpts(1) = -1.
  xpts(2) =  0.
  xpts(3) =  1.

  ! Sparse matrix assembly
  inz = 0
  inz2 = 0

  do ipt = 1, nterm ! Flattened loop over local points to evaluate
     call expandindex(nx, nterm_list, ipt, ipt_list)
     
     do iterm = 1, nterm ! Flattened loop over terms
        call expandindex(nx, nterm_list, iterm, iterm_list)

        prod = 1.
        do ix = 1, nx ! Loop over variables to assemble the term
           pow = iterm_list(ix) - 1
           xval = xpts(ipt_list(ix))
           prod = prod * xval ** pow
        end do

        do ielem = 1, nelem ! Flattened loop over elements
           call expandindex(nx, nelem_list, ielem, ielem_list)

           do ix = 1, nx ! Get variable-wise index in unique ordering
              call uniqueindex(ielem_list(ix), ipt_list(ix), iuniq_list(ix))
           end do
           call contractindex(nx, nuniq_list, iuniq_list, iuniq)

           inz = inz + 1
           data(inz) = prod
           rows(inz) = (ielem - 1) * nterm + ipt - 1
           cols(inz) = (ielem - 1) * nterm + iterm - 1

           if (iterm .eq. 1) then
              inz2 = inz2 + 1
              rows2(inz2) = (ielem - 1) * nterm + ipt - 1
              cols2(inz2) = iuniq - 1
           end if
        end do
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error: incorrect nnz in continuityjacobian', inz, nnz
  end if
  if (inz2 .ne. nnz2) then
     print *, 'Error: incorrect nnz2 in continuityjacobian', inz2, nnz2
  end if

end subroutine tpscontjac



subroutine tpssmthjac(nnz, nnz2, nx, nelems, &
     data, rows, cols, rows2, cols2)

  implicit none

  !f2py intent(in) nnz, nnz2, nx, nelems
  !f2py intent(out) data, rows, cols, rows2, cols2
  !f2py depend(nx) nelems
  !f2py depend(nnz) data, rows, cols
  !f2py depend(nnz2) rows2, cols2

  ! Input
  integer, intent(in) :: nnz, nnz2, nx, nelems(nx)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)
  integer, intent(out) :: rows2(nnz2), cols2(nnz2)

  ! Working
  integer :: nelem, nterm, nuniq
  integer :: nelem_list(nx), nterm_list(nx)
  integer :: nuniq_list(nx), nslice_list(nx)
  integer :: nallslice_list(nx)
  integer :: inz, inz2, ipt, iterm, kx, ix, ielem, islice
  integer :: ipt_list(nx), iterm_list(nx)
  integer :: ielem_list(nx), islice_list(nx), iuniq_list(nx)
  integer :: pow
  double precision :: xpts(3), gpts(3)
  double precision :: prod, xval

  nelem_list(:) = nelems
  nterm_list(:) = 3
  nuniq_list(:) = 1 + 2 * nelem_list

  nelem = product(nelem_list)
  nterm = product(nterm_list)
  nuniq = product(nuniq_list)

  xpts(1) = -1.
  xpts(2) =  0.
  xpts(3) =  1.

  gpts(1) = -sqrt(3./5.)
  gpts(2) =  0.0
  gpts(3) =  sqrt(3./5.)

  nallslice_list = product(nelem_list * nterm_list) &
       / (nelem_list * nterm_list) * nuniq_list

  ! Sparse matrix assembly
  inz = 0
  inz2 = 0

  do ipt = 1, nterm ! Flattened loop over local points to evaluate
     call expandindex(nx, nterm_list, ipt, ipt_list)

     do iterm = 1, nterm ! Flattened loop over terms
        call expandindex(nx, nterm_list, iterm, iterm_list)

        do kx = 1, nx ! Loop over directions

           prod = 1.
           do ix = 1, nx ! Loop over variables to assemble the term
              pow = iterm_list(ix) - 1
              if (ix .ne. kx) then
                 xval = gpts(ipt_list(ix))
                 prod = prod * xval ** pow
              else
                 if (pow .ge. 1) then
                    xval = xpts(ipt_list(ix))
                    prod = prod * pow * xval ** (pow-1)
                 else
                    prod = 0.
                 end if
              end if
           end do
           
           do ielem = 1, nelem ! Flattened loop over elements
              call expandindex(nx, nelem_list, ielem, ielem_list)
              
              inz = inz + 1
              data(inz) = prod
              rows(inz) = (kx - 1) * nelem * nterm + (ielem - 1) * nterm + ipt - 1
              cols(inz) = (ielem - 1) * nterm + iterm - 1
              
              if (iterm .eq. 1) then
                 call uniqueindex(ielem_list(kx), ipt_list(kx), iuniq_list(kx))
                 islice_list = (ielem_list-1) * nterm_list + ipt_list
                 islice_list(kx) = iuniq_list(kx)
                 nslice_list = nelem_list * nterm_list
                 nslice_list(kx) = nuniq_list(kx)
                 call contractindex(nx, nslice_list, islice_list, islice)
                 
                 inz2 = inz2 + 1
                 rows2(inz2) = (kx - 1) * nelem * nterm + (ielem - 1) * nterm + ipt - 1
                 cols2(inz2) = sum(nallslice_list(:kx-1)) + islice - 1
              end if
           end do
        end do
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error: incorrect nnz in smoothnessjacobian', inz, nnz
  end if
  if (inz2 .ne. nnz2) then
     print *, 'Error: incorrect nnz2 in smoothnessjacobian', inz2, nnz2
  end if

end subroutine tpssmthjac




subroutine tpssmthjac2(nnz, nnz2, nx, nelems, &
     data, rows, cols, rows2, cols2)

  implicit none

  !f2py intent(in) nnz, nnz2, nx, nelems
  !f2py intent(out) data, rows, cols, rows2, cols2
  !f2py depend(nx) nelems
  !f2py depend(nnz) data, rows, cols
  !f2py depend(nnz2) rows2, cols2

  ! Input
  integer, intent(in) :: nnz, nnz2, nx, nelems(nx)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)
  integer, intent(out) :: rows2(nnz2), cols2(nnz2)

  ! Working
  integer :: nelem, nterm, nterm2, nuniq
  integer :: nelem_list(nx), nterm_list(nx), nterm2_list(nx)
  integer :: nuniq_list(nx), nslice_list(nx)
  integer :: nallslice_list(nx)
  integer :: inz, inz2, ipt, iterm, kx, ix, ielem, islice
  integer :: ipt_list(nx), iterm_list(nx)
  integer :: ielem_list(nx), islice_list(nx), iuniq_list(nx)
  integer :: pow
  double precision :: xpts(2), gpts(2)
  double precision :: prod, xval

  nelem_list(:) = nelems
  nterm_list(:) = 3
  nterm2_list(:) = 2
  nuniq_list(:) = 1 + nelem_list

  nelem = product(nelem_list)
  nterm = product(nterm_list)
  nterm2 = product(nterm2_list)
  nuniq = product(nuniq_list)

  xpts(1) = -1.
  xpts(2) =  1.

  gpts(1) = -sqrt(1./3.)
  gpts(2) =  sqrt(1./3.)

  nallslice_list = product(nelem_list * nterm2_list) &
       / (nelem_list * nterm2_list) * nuniq_list

  ! Sparse matrix assembly
  inz = 0
  inz2 = 0

  do ipt = 1, nterm2 ! Flattened loop over local points to evaluate
     call expandindex(nx, nterm2_list, ipt, ipt_list)

     do iterm = 1, nterm ! Flattened loop over terms
        call expandindex(nx, nterm_list, iterm, iterm_list)

        do kx = 1, nx ! Loop over directions

           prod = 1.
           do ix = 1, nx ! Loop over variables to assemble the term
              pow = iterm_list(ix) - 1
              if (ix .ne. kx) then
                 xval = gpts(ipt_list(ix))
                 prod = prod * xval ** pow
              else
                 if (pow .ge. 1) then
                    xval = xpts(ipt_list(ix))
                    prod = prod * pow * xval ** (pow-1)
                 else
                    prod = 0.
                 end if
              end if
           end do
           
           do ielem = 1, nelem ! Flattened loop over elements
              call expandindex(nx, nelem_list, ielem, ielem_list)
              
              inz = inz + 1
              data(inz) = prod
              rows(inz) = (kx - 1) * nelem * nterm2 + (ielem - 1) * nterm2 + ipt - 1
              cols(inz) = (ielem - 1) * nterm + iterm - 1
              
              if (iterm .eq. 1) then
                 !call uniqueindex(ielem_list(kx), ipt_list(kx), iuniq_list(kx))
                 iuniq_list(kx) = (ielem_list(kx)-1) + ipt_list(kx)
                 islice_list = (ielem_list-1) * nterm2_list + ipt_list
                 islice_list(kx) = iuniq_list(kx)
                 nslice_list = nelem_list * nterm2_list
                 nslice_list(kx) = nuniq_list(kx)
                 call contractindex(nx, nslice_list, islice_list, islice)
                 
                 inz2 = inz2 + 1
                 rows2(inz2) = (kx - 1) * nelem * nterm2 + (ielem - 1) * nterm2 + ipt - 1
                 cols2(inz2) = sum(nallslice_list(:kx-1)) + islice - 1
              end if
           end do
        end do
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error: incorrect nnz in smoothnessjacobian', inz, nnz
  end if
  if (inz2 .ne. nnz2) then
     print *, 'Error: incorrect nnz2 in smoothnessjacobian', inz2, nnz2
  end if

end subroutine tpssmthjac2




subroutine tpsjacsq(ider, kx, order, nnz, nx, ny, nelems, neval, nrhs, &
     xlimits, xeval, yeval, data, rows, cols, rhs)

  implicit none

  !f2py intent(in) ider, kx, order, nnz, nx, ny, nelems, neval, nrhs, xlimits, xeval, yeval
  !f2py intent(out) data, rows, cols, rhs
  !f2py depend(nx) nelems, xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(neval, ny) yeval
  !f2py depend(nnz) data, rows, cols
  !f2py depend(nrhs, ny) rhs

  ! Input
  integer, intent(in) :: ider, kx, order, nnz, nx, ny, nelems(nx), neval, nrhs
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
  nterm_list(:) = order

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
           else if (ider .eq. 1) then
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
     print *, 'Error: incorrect nnz in approxjacobian', inz, nnz
  end if

  deallocate(prod)

end subroutine tpsjacsq



subroutine tpscubiclocal(nx, nmat, mat)

  implicit none

  !f2py intent(in) nx, nmat
  !f2py intent(out) mat
  !f2py depend(nmat) mat

  ! Input
  integer, intent(in) :: nx, nmat

  ! Output
  double precision, intent(out) :: mat(nmat, nmat)

  ! Working
  integer :: iterm1, iterm2, iterm1_list(nx), iterm2_list(nx)
  integer :: nterm, nterm_list(nx)
  double precision :: prod, xval
  logical :: deriv
  integer :: ix, pow

  nterm_list(:) = 4
  nterm = product(nterm_list)

  mat(:, :) = 0.

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
              print *, 'Error in cubic matrix'
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

        mat(iterm1, iterm2) = prod
     end do
  end do

end subroutine tpscubiclocal



subroutine tpscubicglobalinv(nnz, nx, nmat, nelements, mat, data, rows, cols)

  implicit none

  !f2py intent(in) nnz, nx, nmat, nelements, mat
  !f2py intent(out) data, rows, cols
  !f2py depend(nx) nelements
  !f2py depend(nmat) mat
  !f2py depend(nnz) data, rows, cols

  ! Input
  integer, intent(in) :: nnz, nx, nmat, nelements(nx)
  double precision, intent(in) :: mat(nmat, nmat)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)

  ! Working
  integer :: ielem, iterm1, iterm2, ielem_list(nx), iterm1_list(nx), iterm2_list(nx)
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

     do iterm1 = 1, nterm
        call expandindex(nx, nterm_list, iterm1, iterm1_list)

        do iterm2 = 1, nterm
           call expandindex(nx, nterm_list, iterm2, iterm2_list)

           do ix = 1, nx
              ider_list(ix) = der_map(iterm2_list(ix))
              isid_list(ix) = sid_map(iterm2_list(ix))
              iuniq_list(ix) = (ielem_list(ix) - 1) + isid_list(ix)
           end do
           call contractindex(nx, ndofs_list, ider_list, ider)
           call contractindex(nx, nuniq_list, iuniq_list, iuniq)           

           inz = inz + 1
           data(inz) = mat(iterm1, iterm2)
           rows(inz) = (ielem - 1) * nterm + iterm1 - 1
           cols(inz) = (ider - 1) * nuniq + iuniq - 1
        end do
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error in cubicglobalinvmap', inz, nnz
     call exit(1)
  end if

end subroutine tpscubicglobalinv



subroutine tpscubicmap(nnz, nx, nelements, data, rows, cols)

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
     print *, 'Error in tpscubicmap', inz, nnz
     call exit(1)
  end if

end subroutine tpscubicmap



subroutine tpsquadraticlocal(nx, nmat, mat)

  implicit none

  !f2py intent(in) nx, nmat
  !f2py intent(out) mat
  !f2py depend(nmat) mat

  ! Input
  integer, intent(in) :: nx, nmat

  ! Output
  double precision, intent(out) :: mat(nmat, nmat)

  ! Working
  integer :: ix, ipt, iterm, ipt_list(nx), iterm_list(nx)
  integer :: nterm, nterm_list(nx)
  double precision :: xvec(nx), xpts(3)

  nterm_list(:) = 3
  nterm = product(nterm_list)

  xpts(1) = -1.
  xpts(2) =  0.
  xpts(3) =  1.

  mat(:, :) = 0.

  do ipt = 1, nterm
     call expandindex(nx, nterm_list, ipt, ipt_list)

     do iterm = 1, nterm
        call expandindex(nx, nterm_list, iterm, iterm_list)

        do ix = 1, nx
           xvec(ix) = xpts(ipt_list(ix))
        end do
        mat(ipt, iterm) = product(xvec ** (iterm_list - 1))
     end do
  end do

end subroutine tpsquadraticlocal



subroutine tpsquadraticmap(nnz, nx, nelements, data, rows, cols)

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
  integer :: ielem, ipt, iuniq, ielem_list(nx), ipt_list(nx), iuniq_list(nx)
  integer :: inz, ix
  integer :: nelem, nterm, nuniq, nelem_list(nx), nterm_list(nx), nuniq_list(nx)

  nelem_list(:) = nelements
  nterm_list(:) = 3
  nuniq_list(:) = 1 + 2 * nelements

  nelem = product(nelem_list)
  nterm = product(nterm_list)
  nuniq = product(nuniq_list)

  inz = 0
  do ielem = 1, nelem
     call expandindex(nx, nelem_list, ielem, ielem_list)

     do ipt = 1, nterm
        call expandindex(nx, nterm_list, ipt, ipt_list)

        do ix = 1, nx ! Get variable-wise index in unique ordering
           call uniqueindex(ielem_list(ix), ipt_list(ix), iuniq_list(ix))
        end do
        call contractindex(nx, nuniq_list, iuniq_list, iuniq)
        
        inz = inz + 1
        rows(inz) = (ielem - 1) * nterm + ipt - 1
        cols(inz) = iuniq - 1
     end do
  end do
  data(:) = 1.0

  if (inz .ne. nnz) then
     print *, 'Error in tpsquadraticmap', inz, nnz
     call exit(1)
  end if

end subroutine tpsquadraticmap



subroutine tpsexternalpts(nx, neval, xlimits, xeval, isexternal, isexternal2)

  implicit none

  !f2py intent(in) nx, neval, xlimits, xeval
  !f2py intent(out) isexternal, isexternal
  !f2py depend(nx) xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(neval) isexternal
  !f2py depend(neval, nx) isexternal2

  ! Input
  integer, intent(in) :: nx, neval
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx)

  ! Output
  logical, intent(out) :: isexternal(neval), isexternal2(neval, nx)

  ! Working
  integer :: ieval, ix

  isexternal(:) = .False.
  isexternal2(:, :) = .False.

  do ieval = 1, neval
     do ix = 1, nx
        if (xeval(ieval, ix) .lt. xlimits(ix, 1)) then
           isexternal(ieval) = .True.
           isexternal2(ieval, ix) = .True.
        else if (xeval(ieval, ix) .gt. xlimits(ix, 2)) then
           isexternal(ieval) = .True.
           isexternal2(ieval, ix) = .True.
        end if
     end do
  end do

end subroutine tpsexternalpts



subroutine tpsextrapolation(nx, neval, ndx, xlimits, xeval, dx, dx2)

  implicit none

  !f2py intent(in) nx, neval, ndx, xlimits, xeval
  !f2py intent(out) dx, dx2
  !f2py depend(nx) xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(ndx, nx) dx, dx2

  ! Input
  integer, intent(in) :: nx, neval, ndx
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx)

  ! Output
  double precision, intent(out) :: dx(ndx, nx), dx2(ndx, nx, nx)

  ! Working
  integer :: ieval, ix, iterm, nterm, index
  double precision :: work(neval, nx)

  work(:, :) = xeval(:, :)

  do ieval = 1, neval
     do ix = 1, nx
        work(ieval, ix) = max(xlimits(ix, 1), work(ieval, ix))
        work(ieval, ix) = min(xlimits(ix, 2), work(ieval, ix))
        work(ieval, ix) = xeval(ieval, ix) - work(ieval, ix)
     end do
  end do

  nterm = ndx / neval
  do ieval = 1, neval
     do iterm = 1, nterm
        index = (ieval-1)*nterm + iterm
        dx(index, :) = work(ieval, :)
        do ix = 1, nx
           dx2(index, :, ix) = work(ieval, :)
        end do
        do ix = 1, nx
           dx2(index, ix, :) = dx2(index, ix, :) * work(ieval, :)
        end do
     end do
  end do

end subroutine tpsextrapolation



subroutine tpsjacx(ix, jx, order, nnz, nx, neval, nelements, xlimits, xeval, &
     data, rows, cols)

  implicit none

  !f2py intent(in) ix, jx, order, nnz, nx, neval, nelements, xlimits, xeval
  !f2py intent(out) data, rows, cols
  !f2py depend(nx) nelements, xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(nnz) data, rows, cols

  ! Input
  integer, intent(in) :: ix, jx, order, nnz, nx, neval
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
  nterm_list(:) = order

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
     print *, 'Error in tpsjacx', inz, nnz
     call exit(1)
  end if

end subroutine tpsjacx



subroutine tpsjacxc(cx, ix, jx, order, nnz, nx, neval, nelements, xlimits, xeval, &
     data, rows, cols)

  implicit none

  !f2py intent(in) cx, ix, jx, order, nnz, nx, neval, nelements, xlimits, xeval,
  !f2py intent(out) data, rows, cols
  !f2py depend(nx) nelements, xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(nnz) data, rows, cols

  ! Input
  integer, intent(in) :: cx, ix, jx, order, nnz, nx, neval
  integer, intent(in) :: nelements(nx)
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx)

  ! Output
  double precision, intent(out) :: data(nnz)
  integer, intent(out) :: rows(nnz), cols(nnz)

  ! Working
  integer :: ieval, inz, kx
  integer :: ielem, iterm, ielem_list(nx), iterm_list(nx)
  integer :: nelem, nterm, nelem_list(nx), nterm_list(nx)
  double precision :: bma_d2(nx), dxb_dx(nx), h
  complex*16 :: prod, xbar(nx), cxeval(neval, nx)
  integer :: pow

  h = 1.e-15

  nelem_list(:) = nelements
  nterm_list(:) = order

  nelem = product(nelem_list)
  nterm = product(nterm_list)

  bma_d2 = (xlimits(:, 2) - xlimits(:, 1)) / nelem_list / 2.
  dxb_dx = 1 / bma_d2

  do ieval = 1, neval
     do kx = 1, nx
        cxeval(ieval, kx) = cmplx(xeval(ieval, kx), 0., kind(1.d0))
     end do
     cxeval(ieval, cx) = cxeval(ieval, cx) + cmplx(0., h, kind(1.d0))
  end do

  inz = 0
  do ieval = 1, neval
     do kx = 1, nx
        call findintervalc(nelem_list(kx), xlimits(kx, :), &
             cxeval(ieval, kx), ielem_list(kx), xbar(kx))
     end do
     call contractindex(nx, nelem_list, ielem_list, ielem)
!     print *, 'aaaaaaa', xbar(:)

     do iterm = 1, nterm
        call expandindex(nx, nterm_list, iterm, iterm_list)

        prod = cmplx(1., 0., kind(1.d0))
        do kx = 1, nx
           pow = iterm_list(kx) - 1
           if ((kx .ne. ix) .and. (kx .ne. jx)) then
              prod = prod * xbar(kx) ** pow
           else if ((kx .eq. ix) .and. (kx .eq. jx)) then
              if (pow .ge. 2) then
                 prod = prod * pow * (pow-1) * xbar(kx) ** (pow-2) * dxb_dx(kx) * dxb_dx(kx)
              else
                 prod = cmplx(0., 0., kind(1.d0))
              end if
           else
              if (pow .ge. 1) then
                 prod = prod * pow * xbar(kx) ** (pow-1) * dxb_dx(kx)
              else
                 prod = cmplx(0., 0., kind(1.d0))
              end if
           end if
        end do
!        print *, 'xxxxxx', prod, dimag(prod), xbar(2), xbar(2)**2

        inz = inz + 1
        data(inz) = dimag(prod) / h
        rows(inz) = ieval - 1
        cols(inz) = (ielem - 1) * nterm + iterm - 1
     end do
  end do

  if (inz .ne. nnz) then
     print *, 'Error in tpsjacxc', inz, nnz
     call exit(1)
  end if

end subroutine tpsjacxc
