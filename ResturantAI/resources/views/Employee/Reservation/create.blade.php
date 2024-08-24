<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no" name="viewport">
    <title>Create Reservation</title>

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

                    <div class="col-10 col-md-6 col-lg-12">
                        <div class="card">
                            <form method="POST"
                                action="{{ route('employee.reservation.store') }}"
                                enctype="multipart/form-data">
                                @csrf
                                <div class="card-header">
                                    <h4>Create Reservation</h4>
                                </div>
                                <div class="card-body">
                                    <div class="form-group">
                                        <div class="row">
                                            <div class="col">
                                                <label>Date</label>
                                                <input type="date" class="form-control" name="date" id="dateInput" required="">
                                            </div>

                                            <div class="col">
                                                <label>Day</label>
                                                <input type="text" class="form-control" name="day" id="dayInput" readonly>
                                            </div>

                                            <div class="col">
                                                <label>Time Start</label>
                                                <input type="time" class="form-control" name="timeStart" required="">
                                            </div>
                                        </div>

                                        <br>

                                        <div class="row">

                                            <div class="col-4">
                                                <label>Time End</label>
                                                <input type="time" class="form-control" name="timeEnd" required="">
                                            </div>

                                            <div class="col-4">
                                                <label>Tables</label>
                                                <select class="form-control select2" name="tableID" required>
                                                    @foreach ($tables as $table)
                                                    <option value="{{$table->id}}">{{$table->name}} - {{$table->number}}</option> 
                                                    @endforeach
                                                 
                                                </select>
                                            </div>

                                            <div class="col-4">
                                                <label>Users</label>
                                                <select class="form-control select2" name="userID" required>
                                                    @foreach ($users as $user)
                                                    <option value="{{$user->id}}">{{$user->name}} - {{$user->email}}</option> 
                                                    @endforeach
                                                 
                                                </select>
                                            </div>

                                        </div>

                                        <br>


                                    </div>

                                </div>
                                <div class="card-footer text-center">
                                    <button class="btn btn-primary" type="submit">Create</button>
                                </div>
                            </form>
                        </div>
                    </div>

                </section>
            </div>

        </div>
    </div>

    @include('layouts.Employee.LinkJS')

    <script>
        document.getElementById('dateInput').addEventListener('change', function() {
            var date = new Date(this.value);
            var days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
            var day = days[date.getUTCDay()];
            document.getElementById('dayInput').value = day;
        });
    </script>
</body>

</html>
