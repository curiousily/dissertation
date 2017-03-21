package rl.example.com.myapplication;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.lang.reflect.Method;

/**
 * Created by Venelin Valkov <venelin@curiousily.com>
 * on 3/21/17.
 */

public class ReportWriterReceiver extends BroadcastReceiver {

    public static final String ACTION_WRITE_REPORT = "rl.example.com.myapplication.intent.action.WRITE_REPORT";

    private static final String TAG = "ReportWriter";

    @Override
    public void onReceive(Context context, Intent intent) {
        // use reflection to call emma dump coverage method, to avoid
        // always statically compiling against emma jar
        Log.d("StorageSt", Environment.getExternalStorageState());
        String coverageFilePath = Environment.getExternalStorageDirectory() + File.separator + "coverage.exec";
        File coverageFile = new File(coverageFilePath);
        try {
            coverageFile.createNewFile();
            Class<?> emmaRTClass = Class.forName("com.vladium.emma.rt.RT");
            Method dumpCoverageMethod = emmaRTClass.getMethod("dumpCoverageData",
                    coverageFile.getClass(), boolean.class, boolean.class);

            dumpCoverageMethod.invoke(null, coverageFile, false, false);
            Log.d(TAG, "generateCoverageReport: ok");
        } catch (Exception e) {
            throw new RuntimeException("Is emma jar on classpath?", e);
        }
    }
}