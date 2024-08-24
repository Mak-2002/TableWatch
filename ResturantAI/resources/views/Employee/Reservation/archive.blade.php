<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Reservation Archive Table</title>

    @include('layouts.Employee.LinkHeader')

</head>

<body>
    <div id="app">
        <div class="main-wrapper main-wrapper-1">
            <div class="navbar-bg"></div>

            @include('layouts.Employee.Header')

            <div class="main-sidebar sidebar-style-2">


                @include('layouts.Employee.Sidebar')
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <section class="section">

                    <div class="row">
                        <div class="col-12">
                            <div class="card">
                                <div class="card-header">
                                    <h4>Reservation Archive Table</h4>
                                </div>

                                {{-- message Section --}}

                                @if (session('success_message'))
                                    <div class="alert alert-success">
                                        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                                            <span aria-hidden="true">&times;</span>
                                        </button>
                                        {{ session('success_message') }}
                                    </div>
                                @endif

                                @if (session('error_message'))
                                    <div class="alert alert-danger">
                                        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                                            <span aria-hidden="true">&times;</span>
                                        </button>
                                        {{ session('error_message') }}
                                    </div>
                                @endif
                                {{-- end  message Section --}}
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table class="table table-striped" id="table-1">
                                            <thead>
                                                <tr>
                                                    <th class="text-center">
                                                        #
                                                    </th>
                                                    <th>Day</th>
                                                    <th>Date</th>
                                                    <th>Time Start</th>
                                                    <th>Time End</th>
                                                    <th>Status</th>
                                                    <th>Client Name</th>
                                                    <th>Client Email</th>
                                                    <th>Client Phone</th>
                                                    <th>Table Name</th>
                                                    <th>Table Number</th>
                                                    <th>Table Capacity</th>
                                                    <th>Created By</th>
                                                    <th>Created Date</th>
                                                    <th>Last Update Date</th>
                                                    <th>Deleted Date</th>
                                                    <th>Action</th>
                                                </tr>
                                            </thead>

                                            <tbody>
                                                @foreach ($reservations as $reservation)
                                                    <tr>
                                                        <td>
                                                            {{ $reservation->id }}
                                                        </td>
                                                        <td>{{ $reservation->day }}</td>
                                                        <td>

                                                            {{ $reservation->date }}
                                                        </td>
                                                        <td>{{ $reservation->timeStart }}</td>

                                                        <td>{{ $reservation->timeEnd }}</td>


                                                        <td>
                                                            @if ($reservation->status == 0)
                                                                <div class="badge badge-primary">Not Attended</div>
                                                            @elseif ($reservation->status == 1)
                                                                <div class="badge badge-success"> Attended</div>
                                                            @elseif ($reservation->status == 2)
                                                                <div class="badge badge-danger">Canceled</div>
                                                            @endif

                                                        </td>
                                                        <td>{{ $reservation->user->name }}</td>
                                                        <td>{{ $reservation->user->email }}</td>
                                                        <td>{{ $reservation->user->phone }}</td>
                                                        <td>{{ $reservation->table->name }}</td>
                                                        <td>{{ $reservation->table->number }}</td>
                                                        <td>{{ $reservation->table->capacity }}</td>


                                                        <td>{{ $reservation->employee->name ?? '-' }}</td>
                                                        <td>
                                                            {{ $reservation->created_at }}
                                                        </td>
                                                        <td>
                                                            {{ $reservation->updated_at }}
                                                        </td>
                                                        <td>
                                                            {{ $reservation->deleted_at }}
                                                        </td>
                                                        <td>
                                                            <div class="btn-group">
                                                                <button type="button"
                                                                    class="btn btn-secondary dropdown-toggle"
                                                                    data-toggle="dropdown" aria-haspopup="true"
                                                                    aria-expanded="false">
                                                                    <span class="visually-hidden">Detail</span>
                                                                </button>
                                                                <div class="dropdown-menu">
                                                                    <a class="dropdown-item"
                                                                        href="{{ route('employee.reservation.restore', $reservation->id) }}"
                                                                        style="size: 20px;">Restore</a>

                                                                    <form
                                                                        action="{{ route('employee.reservation.forceDelete', $reservation->id) }}"
                                                                        method="POST">
                                                                        @csrf
                                                                        @method('delete')
                                                                        <button class="dropdown-item" type="submit"
                                                                            style="color: red">Delete</button>
                                                                    </form>

                                                                </div>
                                                            </div>
                                                        </td>

                                                    </tr>
                                                @endforeach

                                            </tbody>

                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                </section>
            </div>

        </div>
    </div>

    @include('layouts.Employee.LinkJS')
</body>

</html>
