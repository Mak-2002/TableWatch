<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Order Invoice Table</title>

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
                                    <h4>Order Invoice Table</h4>
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
                                                    <th>Type Payment</th>
                                                    <th>Tax</th>
                                                    <th>Total Ammount</th>
                                                    <th>Status</th>
                                                    <th>Waiter Name</th>
                                                    <th>Waiter Email</th>
                                                    <th>Waiter Image</th>
                                                    <th>Reservation Day</th>
                                                    <th>Reservation Date</th>
                                                    <th>Reservation Time Start</th>
                                                    <th>Reservation Time End</th>
                                                    <th>Table Name</th>
                                                    <th>Table Number</th>
                                                    <th>Table Capacity</th>
                                                    <th>User Name</th>
                                                    <th>User Email</th>
                                                    <th>User Phone</th>
                                                    <th>Created By</th>
                                                    <th>Created Date</th>
                                                    <th>Last Update Date</th>
                                                    <th>Action</th>
                                                </tr>
                                            </thead>

                                            <tbody>
                                                @foreach ($orderInvoices as $orderInvoice)
                                                    <tr>
                                                        <td>
                                                            {{ $orderInvoice->id }}
                                                        </td>
                                                        <td>{{ $orderInvoice->typePayment ?? '-' }}</td>
                                                        <td>{{ $orderInvoice->tax ?? '-' }}</td>
                                                        <td>{{ $orderInvoice->totalAmmount }}</td>
                                                        <td>
                                                            @if ($orderInvoice->status == 0)
                                                                <div class="badge badge-success">Processing</div>
                                                            @else
                                                                <div class="badge badge-danger">Finished</div>
                                                            @endif

                                                        </td>

                                                        <td>{{ $orderInvoice->waiter->name ?? '-'}}</td>
                                                        <td>{{ $orderInvoice->waiter->email ?? '-'}}</td>
                                                        <td>
                                                            @if($orderInvoice->waiter)
                                                                <img src="{{ $orderInvoice->waiter->img ? asset('image/' . $orderInvoice->waiter->img) : '-' }}"
                                                                     style="width: 100px; height: 100px;">
                                                            @else
                                                                -
                                                            @endif
                                                        </td>
                                                        
                                                        <td>{{ $orderInvoice->reservation->day }}</td>
                                                        <td>{{ $orderInvoice->reservation->date }}</td>
                                                        <td>{{ $orderInvoice->reservation->timeStart }}</td>
                                                        <td>{{ $orderInvoice->reservation->timeEnd }}</td>

                                                        <td>{{ $orderInvoice->reservation->table->name }}</td>
                                                        <td>{{ $orderInvoice->reservation->table->number }}</td>
                                                        <td>{{ $orderInvoice->reservation->table->capacity }}</td>

                                                        <td>{{ $orderInvoice->reservation->user->name }}</td>
                                                        <td>{{ $orderInvoice->reservation->user->email }}</td>
                                                        <td>{{ $orderInvoice->reservation->user->phone }}</td>


                                                        <td>{{ $orderInvoice->employee->name }}</td>
                                                        <td>{{ $orderInvoice->created_at }}</td>
                                                        <td>{{ $orderInvoice->updated_at }}</td>
                                                        <td>
                                                            @if($orderInvoice->status == 0)

                                                            <div class="btn-group">
                                                                <button type="button"
                                                                    class="btn btn-secondary dropdown-toggle"
                                                                    data-toggle="dropdown" aria-haspopup="true"
                                                                    aria-expanded="false">
                                                                    <span class="visually-hidden">Detail</span>
                                                                </button>
                                                                <div class="dropdown-menu">
                                                                    <a class="dropdown-item"
                                                                        href="{{ route('employee.order.invoice.food.details', $orderInvoice->id) }}"
                                                                        style="size: 20px;">Add Food</a>

                                                                        <form
                                                                        action="{{ route('employee.order.invoice.finish', $orderInvoice->id) }}"
                                                                        method="POST">
                                                                        @csrf
                                                                        @method('put')
                                                                        <button class="dropdown-item" type="submit"
                                                                            style="color: rgb(33, 100, 255)">Finish Order Invoice</button>
                                                                    </form>

                                                                    <form
                                                                        action="{{ route('employee.order.invoice.delete', $orderInvoice->id) }}"
                                                                        method="POST">
                                                                        @csrf
                                                                        @method('delete')
                                                                        <button class="dropdown-item" type="submit"
                                                                            style="color: red">Delete</button>
                                                                    </form>

                                                                </div>
                                                            </div>
                                                                                                                            
                                                            @endif
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
