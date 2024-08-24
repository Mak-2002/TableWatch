<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Order Invoice Details Table</title>

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
                                    <h4>Order Invoice Details Table</h4>
                                    <form action="{{ route('employee.order.invoice.add.food' , $orderInvoiceID)}}" method="GET">
                                        @csrf
                                        <button type="submit" class="btn btn-primary">
                                            <span>Add Food</span>
                                        </button>
                                    </form>
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
                                                    <th>Food Quantity</th>
                                                    <th>Food Note</th>
                                                    <th>Food Ammount</th>
                                                    <th>Food Name</th>
                                                    <th>Food Details</th>
                                                    <th>Food Note</th>
                                                    <th>Food Image</th>
                                                    <th>Food Price</th>
                                                    <th>Category Name</th>
                                                    <th>Category Type</th>
                                                    <th>Created Date</th>
                                                    <th>Last Update Date</th>
                                                    <th>Action</th>
                                                </tr>
                                            </thead>

                                            <tbody>
                                                @foreach ($orderInvoiceDetails as $orderInvoiceDetail)
                                                    <tr>
                                                        <td>
                                                            {{ $orderInvoiceDetail->id }}
                                                        </td>
                                                        <td>{{ $orderInvoiceDetail->foodQuantity }}</td>
                                                        <td>{{ $orderInvoiceDetail->foodNote }}</td>
                                                        <td>{{ $orderInvoiceDetail->foodAmmount }}</td>

                                                        <td>{{ $orderInvoiceDetail->foodCategory->food->name }}</td>
                                                        <td>{{ $orderInvoiceDetail->foodCategory->food->details }}</td>
                                                        <td>{{ $orderInvoiceDetail->foodCategory->food->note }}</td>
                                                        <td><img src="{{ asset('image/' . $orderInvoiceDetail->foodCategory->food->img) }}"
                                                                style="width: 100px; height: 100px;"></td>

                                                        <td>{{ $orderInvoiceDetail->foodCategory->price }}</td>

                                                        <td>{{ $orderInvoiceDetail->foodCategory->category->name }}
                                                        </td>

                                                        <td>
                                                            @if ($orderInvoiceDetail->foodCategory->category->type == 0)
                                                                <div class="badge badge-success">Meal</div>
                                                            @elseif($orderInvoiceDetail->foodCategory->category->type == 1)
                                                                <div class="badge badge-success">Sandwich</div>
                                                            @endif

                                                        </td>
                                                        <td>{{ $orderInvoiceDetail->created_at }}</td>
                                                        <td>{{ $orderInvoiceDetail->updated_at }}</td>
                                                        <td>
                                                            <div class="btn-group">
                                                                <button type="button"
                                                                    class="btn btn-secondary dropdown-toggle"
                                                                    data-toggle="dropdown" aria-haspopup="true"
                                                                    aria-expanded="false">
                                                                    <span class="visually-hidden">Detail</span>
                                                                </button>
                                                                <div class="dropdown-menu">
                                                                    <form
                                                                        action="{{ route('employee.order.invoice.delete.food', $orderInvoiceDetail->id) }}"
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
