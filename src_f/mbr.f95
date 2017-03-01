subroutine knot_vector_uniform(order, num_ctrl, knot_vector)

  implicit none

  !f2py intent(in) order, num_ctrl
  !f2py intent(out) knot_vector
  !f2py depend(order, num_ctrl) knot_vector

  ! Input
  integer, intent(in) :: order, num_ctrl

  ! Output
  double precision, intent(out) :: knot_vector(order + num_ctrl)

  ! Working
  integer :: ind

  do ind = 1, order + num_ctrl
    knot_vector(ind) = 1.0 * (ind - order) / (num_ctrl - order + 1)
  end do

end subroutine knot_vector_uniform



subroutine compute_jac(ix, jx, nx, nP, nB, ks, ms, t, Ba, Bi, Bj)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) ix, jx, nx, nP, nB, ks, ms, t
  !f2py intent(out) Ba, Bi, Bj
  !f2py depend(nx) ks, ms
  !f2py depend(nP,nx) t
  !f2py depend(nB) Ba, Bi, Bj

  !Input
  integer, intent(in) ::  ix, jx, nx, nP, nB, ks(nx), ms(nx)
  double precision, intent(in) ::  t(nP,nx)

  !Output
  double precision, intent(out) ::  Ba(nB)
  integer, intent(out) ::  Bi(nB), Bj(nB)

  !Working
  integer k, m, na
  integer iP, iB, iC
  integer i, kx, ia, ik, i0, rem
  double precision, allocatable, dimension(:) ::  d, B

  na = product(ks)

  do iP=1,nP
     Bi((iP-1)*na+1:iP*na) = iP - 1
  end do

  Ba(:) = 1.0
  Bj(:) = 0
  do kx=1,nx
     k = ks(kx)
     m = ms(kx)
     allocate(d(k+m))
     allocate(B(k))
     call knot_vector_uniform(k, m, d)
     do iP=1,nP
        if ((ix .ne. kx) .and. (jx .ne. kx)) then
           call basis(k, k+m, t(iP, kx), d, B, i0)
        else if ((ix .eq. kx) .and. (jx .ne. kx)) then
           call basis1(k, k+m, t(iP, kx), d, B, i0)
        else if ((ix .ne. kx) .and. (jx .eq. kx)) then
           call basis1(k, k+m, t(iP, kx), d, B, i0)
        else if ((ix .eq. kx) .and. (jx .eq. kx)) then
           call basis2(k, k+m, t(iP, kx), d, B, i0)
        end if
        do ia=1,na
           rem = ia - 1
           do i=nx,kx,-1
              rem = mod(rem,product(ks(:i)))
           end do
           ik = rem/product(ks(:kx-1)) + 1
           iB = (iP-1)*na + ia
           iC = (ik + i0 - 1)*product(ms(:kx-1))
           Ba(iB) = Ba(iB) * B(ik)
           Bj(iB) = Bj(iB) + iC
        end do
     end do
     deallocate(d)
     deallocate(B)
  end do

end subroutine compute_jac



subroutine basis(k, kpm, t, d, B, i0)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) k,kpm,t,d
  !f2py intent(out) B,i0
  !f2py depend(kpm) d
  !f2py depend(k) B

  !Input
  integer, intent(in) ::  k, kpm
  double precision, intent(in) ::  t, d(kpm)

  !Output
  double precision, intent(out) ::  B(k)
  integer, intent(out) ::  i0

  !Working
  integer i, j, j1, j2, l, m, n

  m = kpm - k

  i0 = -1
  do i=k,m
     if ((d(i) .le. t) .and. (t .lt. d(i+1))) then
        i0 = i-k
     end if
  end do

  B(:) = 0
  B(k) = 1

  if (t .eq. d(m+k)) then
     i0 = m-k
  end if

  !if (i0 .eq. -1) then
  !   print *, 'MBI error: t outside of knot spans'
  !end if

  do i=2,k
     l = i-1
     j1 = k-l
     j2 = k
     n = i0 + j1
     if (d(n+l+1) .ne. d(n+1)) then
        B(j1) = (d(n+l+1)-t)/(d(n+l+1)-d(n+1))*B(j1+1)
     else
        B(j1) = 0
     end if
     do j=j1+1,j2-1
        n = i0 + j
        if (d(n+l) .ne. d(n)) then
           B(j) = (t-d(n))/(d(n+l)-d(n))*B(j)
        else
           B(j) = 0
        end if
        if (d(n+l+1) .ne. d(n+1)) then
           B(j) = B(j) + (d(n+l+1)-t)/(d(n+l+1)-d(n+1))*B(j+1)
        end if
     end do
     n = i0 + j2
     if (d(n+l) .ne. d(n)) then
        B(j2) = (t-d(n))/(d(n+l)-d(n))*B(j2)
     else
        B(j2) = 0
     end if
  end do

end subroutine basis


subroutine basis1(k, kpm, t, d, F, i0)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) k,kpm,t,d
  !f2py intent(out) F,i0
  !f2py depend(kpm) d
  !f2py depend(k) F

  !Input
  integer, intent(in) ::  k, kpm
  double precision, intent(in) ::  t, d(kpm)

  !Output
  double precision, intent(out) ::  F(k)
  integer, intent(out) ::  i0

  !Working
  double precision B(k), b1, b2, f1, f2, den
  integer i, j, j1, j2, l, m, n

  m = kpm - k

  i0 = -1
  do i=k,m
     if ((d(i) .le. t) .and. (t .lt. d(i+1))) then
        i0 = i-k
     end if
  end do

  B(:) = 0
  B(k) = 1

  if (t .eq. d(m+k)) then
     i0 = m-k
  end if

  F(:) = 0.0
  do i=2,k
     l = i-1
     j1 = k-l
     j2 = k
     do j=j1,j2
        n = i0 + j
        if (d(n+l) .ne. d(n)) then
           den = d(n+l)-d(n)
           b1 = (t-d(n))/den*B(j)
           f1 = (B(j)+(t-d(n))*F(j))/den
        else
           b1 = 0
           f1 = 0
        end if
        if ((j .ne. j2) .and. (d(n+l+1) .ne. d(n+1))) then
           den = d(n+l+1)-d(n+1)
           b2 = (d(n+l+1)-t)/den*B(j+1)
           f2 = ((d(n+l+1)-t)*F(j+1)-B(j+1))/den
        else
           b2 = 0
           f2 = 0
        end if
        B(j) = b1 + b2
        F(j) = f1 + f2
     end do
  end do

end subroutine basis1


subroutine basis2(k, kpm, t, d, S, i0)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) k,kpm,t,d
  !f2py intent(out) S,i0
  !f2py depend(kpm) d
  !f2py depend(k) S

  !Input
  integer, intent(in) ::  k, kpm
  double precision, intent(in) ::  t, d(kpm)

  !Output
  double precision, intent(out) ::  S(k)
  integer, intent(out) ::  i0

  !Working
  double precision B(k), F(k), b1, b2, f1, f2, s1, s2, den
  integer i, j, j1, j2, l, m, n

  m = kpm - k

  i0 = -1
  do i=k,m
     if ((d(i) .le. t) .and. (t .lt. d(i+1))) then
        i0 = i-k
     end if
  end do

  B(:) = 0
  B(k) = 1

  if (t .eq. d(m+k)) then
     i0 = m-k
  end if

  F(:) = 0.0
  S(:) = 0.0
  do i=2,k
     l = i-1
     j1 = k-l
     j2 = k
     do j=j1,j2
        n = i0 + j
        if (d(n+l) .ne. d(n)) then
           den = d(n+l)-d(n)
           b1 = (t-d(n))/den*B(j)
           f1 = (B(j)+(t-d(n))*F(j))/den
           s1 = (2*F(j)+(t-d(n))*S(j))/den
        else
           b1 = 0
           f1 = 0
           s1 = 0
        end if
        if ((j .ne. j2) .and. (d(n+l+1) .ne. d(n+1))) then
           den = d(n+l+1)-d(n+1)
           b2 = (d(n+l+1)-t)/den*B(j+1)
           f2 = ((d(n+l+1)-t)*F(j+1)-B(j+1))/den
           s2 = ((d(n+l+1)-t)*S(j+1)-2*F(j+1))/den
        else
           b2 = 0
           f2 = 0
           s2 = 0
        end if
        B(j) = b1 + b2
        F(j) = f1 + f2
        if (i .gt. 2) then
           S(j) = s1 + s2
        end if
     end do
  end do

end subroutine basis2



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
