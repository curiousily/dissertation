package rl.example.com.myapplication;

import android.app.Application;
import android.net.wifi.WifiManager;

/**
 * Created by Venelin Valkov <venelin@curiousily.com>
 * on 3/22/17.
 */
public class App extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        CodeCoverageHTTPServer.start((WifiManager) getSystemService(WIFI_SERVICE));
    }
}