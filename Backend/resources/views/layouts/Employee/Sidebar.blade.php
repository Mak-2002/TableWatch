<aside id="sidebar-wrapper">
    <div class="sidebar-brand">
        <a href="index.html">Employee Dashboard</a>
    </div>
    <div class="sidebar-brand sidebar-brand-sm">
        <a href="index.html">ED</a>
    </div>
    <ul class="sidebar-menu">
        <li class="menu-header">Dashboard</li>
        <li class="dropdown active">
            <a href="{{ route('employee.dashboard') }}"><i>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                        class="bi bi-house" viewBox="0 0 16 16">
                        <path
                            d="M8.707 1.5a1 1 0 0 0-1.414 0L.646 8.146a.5.5 0 0 0 .708.708L2 8.207V13.5A1.5 1.5 0 0 0 3.5 15h9a1.5 1.5 0 0 0 1.5-1.5V8.207l.646.647a.5.5 0 0 0 .708-.708L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293zM13 7.207V13.5a.5.5 0 0 1-.5.5h-9a.5.5 0 0 1-.5-.5V7.207l5-5z" />
                    </svg>
                </i><span>Dashboard</span></a>
        </li>
        <li class="menu-header">System</li>
        <li class="dropdown">
            <a href="#" class="nav-link has-dropdown" data-toggle="dropdown">
                <i>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                        class="bi bi-card-list" viewBox="0 0 16 16">
                        <path
                            d="M14.5 3a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5v-9a.5.5 0 0 1 .5-.5zm-13-1A1.5 1.5 0 0 0 0 3.5v9A1.5 1.5 0 0 0 1.5 14h13a1.5 1.5 0 0 0 1.5-1.5v-9A1.5 1.5 0 0 0 14.5 2z" />
                        <path
                            d="M5 8a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 0 1h-7A.5.5 0 0 1 5 8m0-2.5a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 0 1h-7a.5.5 0 0 1-.5-.5m0 5a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 0 1h-7a.5.5 0 0 1-.5-.5m-1-5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0M4 8a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0m0 2.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0" />
                    </svg>

                </i>
                <span>Reservations Managment</span></a>
            <ul class="dropdown-menu">
                <li><a class="nav-link" href="{{ route('employee.reservation.index') }}">Reservations Table</a></li>
                <li><a class="nav-link" href="{{ route('employee.reservation.create') }}">Add Reservation</a></li>
                <li><a class="nav-link" href="{{ route('employee.reservation.archive') }}">Reservations Archive </a>
                </li>
            </ul>
        </li>

        <br>

        <li class="dropdown">
            <a href="#" class="nav-link has-dropdown"><i>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                        class="bi bi-table" viewBox="0 0 16 16">
                        <path
                            d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2zm15 2h-4v3h4zm0 4h-4v3h4zm0 4h-4v3h3a1 1 0 0 0 1-1zm-5 3v-3H6v3zm-5 0v-3H1v2a1 1 0 0 0 1 1zm-4-4h4V8H1zm0-4h4V4H1zm5-3v3h4V4zm4 4H6v3h4z" />
                    </svg>
                </i> <span>Table Managment</span></a>
            <ul class="dropdown-menu">
                <li><a class="nav-link" href="{{ route('employee.table.index') }}">Tables Table</a></li>
                <li><a class="nav-link" href="{{ route('employee.table.create') }}">Add Table</a></li>
                <li><a class="nav-link" href="{{ route('employee.table.archive') }}">Tables Archive</a></li>

        </li>
    </ul>
    </li>

    {{-- <li class="menu-header">System</li> --}}

    <br>

    <li class="dropdown">
        <a href="#" class="nav-link has-dropdown"><i>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                    class="bi bi-person-badge" viewBox="0 0 16 16">
                    <path d="M6.5 2a.5.5 0 0 0 0 1h3a.5.5 0 0 0 0-1zM11 8a3 3 0 1 1-6 0 3 3 0 0 1 6 0" />
                    <path
                        d="M4.5 0A2.5 2.5 0 0 0 2 2.5V14a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V2.5A2.5 2.5 0 0 0 11.5 0zM3 2.5A1.5 1.5 0 0 1 4.5 1h7A1.5 1.5 0 0 1 13 2.5v10.795a4.2 4.2 0 0 0-.776-.492C11.392 12.387 10.063 12 8 12s-3.392.387-4.224.803a4.2 4.2 0 0 0-.776.492z" />
                </svg>
            </i> <span>User
                Manager</span></a>
        <ul class="dropdown-menu">
            <li><a class="nav-link" href="{{ route('employee.user.index') }}">User Table</a></li>
            <li><a class="nav-link" href="{{ route('employee.user.create') }}">Add User</a></li>
            <li><a class="nav-link" href="{{ route('employee.user.archive') }}">User Archive</a></li>
        </ul>
    </li>

    <br>

    <li class="dropdown">
        <a href="#" class="nav-link has-dropdown"><i>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                    class="bi bi-person-circle" viewBox="0 0 16 16">
                    <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0" />
                    <path fill-rule="evenodd"
                        d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1" />
                </svg>
            </i> <span>Waiter
                Manager</span></a>
        <ul class="dropdown-menu">
            <li><a class="nav-link" href="{{ route('employee.waiter.index') }}">Waiter Table</a></li>
            <li><a class="nav-link" href="{{ route('employee.waiter.create') }}">Add Waiter </a></li>
            <li><a class="nav-link" href="{{ route('employee.waiter.archive') }}">Waiter Archive</a></li>
        </ul>
    </li>

    <br>


    <li class="dropdown">
        <a href="#" class="nav-link has-dropdown">
            <i>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                    class="bi bi-egg-fried" viewBox="0 0 16 16">
                    <path d="M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6" />
                    <path
                        d="M13.997 5.17a5 5 0 0 0-8.101-4.09A5 5 0 0 0 1.28 9.342a5 5 0 0 0 8.336 5.109 3.5 3.5 0 0 0 5.201-4.065 3.001 3.001 0 0 0-.822-5.216zm-1-.034a1 1 0 0 0 .668.977 2.001 2.001 0 0 1 .547 3.478 1 1 0 0 0-.341 1.113 2.5 2.5 0 0 1-3.715 2.905 1 1 0 0 0-1.262.152 4 4 0 0 1-6.67-4.087 1 1 0 0 0-.2-1 4 4 0 0 1 3.693-6.61 1 1 0 0 0 .8-.2 4 4 0 0 1 6.48 3.273z" />
                </svg>
            </i>
            <span>Food Manager
            </span></a>
        <ul class="dropdown-menu">
            <li><a href="{{ route('employee.food.index') }}">Food Table</a></li>
            <li><a href="{{ route('employee.food.create') }}">Add Food</a></li>
            <li><a href="{{ route('employee.food.archive') }}">Food Archive</a></li>
        </ul>
    </li>

    <br>

    <li class="dropdown">
        <a href="#" class="nav-link has-dropdown"><i>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                    class="bi bi-egg" viewBox="0 0 16 16">
                    <path
                        d="M8 15a5 5 0 0 1-5-5c0-1.956.69-4.286 1.742-6.12.524-.913 1.112-1.658 1.704-2.164C7.044 1.206 7.572 1 8 1s.956.206 1.554.716c.592.506 1.18 1.251 1.704 2.164C12.31 5.714 13 8.044 13 10a5 5 0 0 1-5 5m0 1a6 6 0 0 0 6-6c0-4.314-3-10-6-10S2 5.686 2 10a6 6 0 0 0 6 6" />
                </svg>
            </i> <span>Food Category Manager</span></a>
        <ul class="dropdown-menu">
            <li><a class="nav-link" href="{{ route('employee.food.category.index') }}">Food Category Table</a></li>
            <li><a class="nav-link" href="{{ route('employee.food.category.create') }}">Add Food Category</a></li>
            <li><a class="nav-link" href="{{ route('employee.food.category.archive') }}">Food Category Archive</a>
            </li>

        </ul>
    </li>

    <br>

    <li class="dropdown">
        <a href="#" class="nav-link has-dropdown"> <i>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                    class="bi bi-border-width" viewBox="0 0 16 16">
                    <path
                        d="M0 3.5A.5.5 0 0 1 .5 3h15a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5H.5a.5.5 0 0 1-.5-.5zm0 5A.5.5 0 0 1 .5 8h15a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H.5a.5.5 0 0 1-.5-.5zm0 4a.5.5 0 0 1 .5-.5h15a.5.5 0 0 1 0 1H.5a.5.5 0 0 1-.5-.5" />
                </svg>
            </i> <span>Order Invoice
                Manager</span></a>
        <ul class="dropdown-menu">
            <li><a href="{{ route('employee.order.invoice.index') }}">Order Invoice Table</a></li>
            <li><a href="{{ route('employee.order.invoice.create') }}">Add Order Invoice</a></li>
            <li><a href="{{ route('employee.order.invoice.archive') }}">Order Invoice Archive</a></li>

        </ul>
    </li>

    </ul>
    <br>
    <br>
</aside>
