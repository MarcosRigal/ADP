package com.example.locationtrack

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import io.realm.kotlin.Realm
import io.realm.kotlin.ext.query
import io.realm.kotlin.internal.platform.runBlocking
import io.realm.kotlin.mongodb.App
import io.realm.kotlin.mongodb.Credentials
import io.realm.kotlin.mongodb.User
import io.realm.kotlin.mongodb.annotations.ExperimentalFlexibleSyncApi
import io.realm.kotlin.mongodb.ext.subscribe
import io.realm.kotlin.mongodb.sync.SyncConfiguration
import io.realm.kotlin.types.RealmObject
import io.realm.kotlin.types.annotations.PrimaryKey
import org.mongodb.kbson.ObjectId


class Location:RealmObject {//Empty constructor required by Realm
    @PrimaryKey
    var _id: ObjectId = ObjectId()
    var ownerId: String = ""
    var accuracy: Float = 0.0F
    var latitude: Double = 0.0
    var longitude: Double = 0.0

    var altitude: Double = 0.0

    var speed: Float = 0.0F
    var time: Long = 0
}

var user: User? = null
var syncedRealm: Realm? = null

class TrackingService : Service() {
    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        buildNotification()
        loginToMongo()
    }

    private fun createNotificationChannel() {
        // Create the NotificationChannel, but only on API 26+ because
        // the NotificationChannel class is new and not in the support library
        val name = getString(R.string.channel_name)
        val descriptionText = getString(R.string.channel_description)
        val importance = NotificationManager.IMPORTANCE_DEFAULT
        val channel = NotificationChannel(CHANNEL_ID.toString(), name, importance).apply {
                description = descriptionText
        }

        // Register the channel with the system
        val notificationManager: NotificationManager =
            getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)

    }

    //Create the persistent notification//
    @SuppressLint("UnspecifiedImmutableFlag")
    private fun buildNotification() {
        val stop = "stop"
        val broadcastIntent:PendingIntent
        val stopR = IntentFilter(stop)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // API33+
            registerReceiver(stopReceiver, stopR, RECEIVER_NOT_EXPORTED)
            broadcastIntent = PendingIntent.getBroadcast(
                this, 0, Intent(stop), PendingIntent.FLAG_IMMUTABLE
            )
        } else {
            // API level 32 and lower
            registerReceiver(stopReceiver, stopR)
            broadcastIntent = PendingIntent.getBroadcast(
                this, 0, Intent(stop), PendingIntent.FLAG_UPDATE_CURRENT
            )
        }


        // Create the persistent notification//
        val builder = Notification.Builder(this, CHANNEL_ID.toString())
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.tracking_enabled_notif))
            //Make this notification ongoing so it can’t be dismissed by the user//
            .setOngoing(true)
            .setContentIntent(broadcastIntent)
            //.addAction(1, "Stop", broadcastIntent)
            .setSmallIcon(R.drawable.ic_tracking_enabled)
        startForeground(1, builder.build())
    }

    //Stop the service on notification press
    private var stopReceiver: BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {

            //Unregister the BroadcastReceiver when the notification is tapped//

            syncedRealm?.close()
            unregisterReceiver(this)

            //Stop the Service//
            stopSelf()
            //Log.d("Stopped", "Service stopped")
            // Optional: Android should manage itself -> no need to use exitProcess(0)

        }
    }

    @OptIn(ExperimentalFlexibleSyncApi::class)
    private fun loginToMongo(){

        //Realm.init(applicationContext)
        val appID = getString(R.string.app_id)
        val app = App.create(appID)

        runBlocking {
            val emailPasswordCredentials: Credentials = Credentials.emailPassword(
                getString(R.string.test_email), getString(R.string.test_password)
            )
            val myuser = app.login(emailPasswordCredentials)
            val config =
                SyncConfiguration.Builder(
                    user = myuser,
                    schema = setOf(Location::class)
                )
                    .initialSubscriptions{
                    }
                    .build()
            val realm = Realm.open(config)
            Log.v("REALM_OPENED","Successfully opened synced realm: ${realm.configuration.name}")
            // Wait for initial subscriptions to sync to Atlas
            // Subscribe to all objects of a specific type
            val realmQuery = realm.query<Location>()
            realmQuery.subscribe()
            user = myuser
            syncedRealm = realm
            requestLocationUpdates2(syncedRealm!!)
        } //TO-DO: manage possible errors

    }


    //Initiate the request to track the device's location//
    private fun requestLocationUpdates2(realm: Realm) {
        val request = LocationRequest.create()

        //https://docs.mongodb.com/realm/sdk/android/examples/mongodb-remote-access/

        // API Ref for Location:
        // https://developer.android.com/reference/android/location/Location?hl=es-419

        //Specify how often your app should request the device’s location//
        request.interval = 15000 //ms
        request.fastestInterval = 9000 //ms

        //Get the most accurate location data available//
        request.priority = LocationRequest.PRIORITY_HIGH_ACCURACY
        val client = LocationServices.getFusedLocationProviderClient(this)
        //val path = getString(R.string.firebase_path)
        val permission = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.ACCESS_FINE_LOCATION)

        //If the app currently has access to the location permission...//
        if (permission == PackageManager.PERMISSION_GRANTED) {

            //...then request location updates//
            client.requestLocationUpdates(request, object : LocationCallback() {

                override fun onLocationResult(locationResult: LocationResult) {

                    var data: Location?

                    val location = locationResult.lastLocation
                    //Save the location data to the database//

                    Log.d("Location", "Location: $location")

                    if (!realm.isClosed()){
                        runBlocking {
                            realm.write {

                                //Creating location data to sync
                                data = Location().apply {
                                    ownerId = user!!.id
                                    accuracy = location!!.accuracy
                                    latitude = location.latitude
                                    longitude = location.longitude

                                    altitude = location.altitude

                                    speed = location.speed
                                    time = location.time
                                }
                                copyToRealm(data!!)
                            }
                        }
                    }
                    else{
                        return //if the realm is closed stop getting location updates
                    }
                }
            }, Looper.getMainLooper())

        }
    }

    companion object {
        private const val CHANNEL_ID = 1945
        //val uid = kotlin.math.abs(Random.nextInt()).toString()
    }

}